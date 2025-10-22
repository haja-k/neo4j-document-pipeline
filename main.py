import time
import re, os, uuid
from typing import List, Optional
from fastapi import FastAPI, Body, UploadFile
from neo4j.exceptions import ServiceUnavailable
from fastapi.middleware.cors import CORSMiddleware
from tasks import ingest_markdown_task
from pydantic import BaseModel, Field

from neo4j_connect import driver
from graphutil import (
    get_question_embedding,
    hybrid_candidates,
    mmr_select,
    fulltext_search,
    diversify_by_document,
    vector_find_similar_nodes,
    traverse_neighbors,
    format_graph_context,
    generate_llm_answer,
    decompose_question,
    DEFAULT_LABELS, 
    driver
)

app = FastAPI(title="GraphRAG API")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---------- Heuristic for AUTO decomposition ----------
def is_complex_question(q: str) -> bool:
    text = (q or "").strip()
    if not text:
        return False
    qm = text.count("?") >= 2
    long_and_conj = (len(text.split()) >= 18) and bool(re.search(r"\b(and|or)\b", text, flags=re.I))
    many_commas = text.count(",") + text.count(";") >= 2
    enumerations = bool(re.search(r"(?:^|\s)(\d+\.)\s", text))
    return qm or long_and_conj or many_commas or enumerations

# ---------- Request schema ----------
class RagBody(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = 7
    hops: int = 1
    labels: Optional[List[str]] = None
    use_hybrid: bool = True
    alpha_vec: float = 0.6
    beta_kw: float = 0.25
    use_mmr: bool = True
    use_cross_doc: bool = True
    decompose: Optional[bool] = False

@app.get("/test")
def test():
    with driver.session() as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    return {"nodes": n}

@app.get("/healthz")
def healthz():
    try:
        with driver.session() as s:
            c = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        return {"ok": True, "nodes": c}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    
@app.on_event("startup")
def verify_embedding_system():
    """Verify embedding system with retry logic for Neo4j connection"""
    max_retries = 10
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            from graphutil import get_question_embedding, driver, DEFAULT_LABELS
            
            print(f"🔧 Attempt {attempt + 1}/{max_retries}: Connecting to Neo4j...")
            
            # Test basic connection first
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    print("✅ Neo4j connection successful")
            
            # Test embedding generation
            test_embedding = get_question_embedding("test question")
            actual_dimensions = len(test_embedding)
            print(f"🔍 Embedding system: {actual_dimensions} dimensions")
            
            if actual_dimensions != 3072:
                print(f"🚨 WARNING: Expected 3072D embeddings but got {actual_dimensions}D")
                print("💡 Solution: Re-ingest all data with correct embedding model")
            
            # Quick health check - try a simple vector search
            with driver.session() as session:
                # Check if any vector indexes exist and work
                result = session.run("""
                SHOW INDEXES WHERE type = 'VECTOR' 
                YIELD name, labelsOrTypes
                RETURN count(*) as index_count
                """).single()
                
                if result and result["index_count"] > 0:
                    print(f"✅ Found {result['index_count']} vector indexes")
                else:
                    print("ℹ️  No vector indexes found - normal if no data ingested yet")
            
            print("✅ Embedding system verification complete")
            return  # Success, exit the function
            
        except ServiceUnavailable as e:
            print(f"⚠️  Neo4j not ready yet (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"🕐 Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                print("❌ Failed to connect to Neo4j after multiple attempts")
                print("💡 Please check if Neo4j container is running and healthy")
                break
                
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            break

@app.post("/graphrag")
def graphrag(body: RagBody = Body(...)):
    if not body.question.strip():
        return {"answer": "Please provide a question.", "facts": "", "seeds": []}

    q0 = body.question.strip()

    # NOTE: Debugging
    print(f"Processing question: {q0}")

    # Decomposition: AUTO when decompose is None; else honor True/False
    if body.decompose is True:
        questions = decompose_question(q0, max_parts=3) or [q0]
    elif body.decompose is False:
        questions = [q0]
    else:
        questions = decompose_question(q0, max_parts=3) or [q0] if is_complex_question(q0) else [q0]

    all_facts: List[str] = []
    all_seeds_meta: List[dict] = []
    answers: List[str] = []                       # collect answers ONCE (no second LLM call)
    labels = body.labels or DEFAULT_LABELS        # compute once for params echo

    for q in questions:
        # 1) Embed question
        qvec = get_question_embedding(q)
        
        # NOTE: Debugging
        print(f"Got embedding for: {q}") 

        # 2) Hybrid candidates (vector + keyword)
        cands = hybrid_candidates(
            question=q, qvec=qvec, labels=labels,
            k_vec=max(12, body.top_k), k_kw=max(12, body.top_k),
            alpha_vec=body.alpha_vec, beta_kw=body.beta_kw
        )

        # NOTE: Debugging
        print(f"Found {len(cands)} candidates")

        # 3) MMR diversification
        if body.use_mmr and len(cands) > body.top_k:
            cands = mmr_select(cands, k=body.top_k)
        else:
            cands = cands[:body.top_k]

        # 4) Cross-document coverage
        if body.use_cross_doc and len(cands) > 1:
            cands = diversify_by_document(cands, k=len(cands))

        # 5) Expand neighbors - with error handling
        seed_ids = [n.element_id for n, _ in cands]
        print(f"Expanding {len(seed_ids)} seeds with {body.hops} hops")
        
        try:
            expanded = traverse_neighbors(
                seed_ids,
                max_hops=max(1, min(body.hops, 3))
            )
            print(f"Expanded graph: {len(expanded.get('nodes', []))} nodes, {len(expanded.get('rels', []))} relationships")
        except Exception as e:
            print(f"Error in traverse_neighbors: {e}")
            expanded = {"nodes": [], "rels": []}

        # 6) Format context & answer
        facts = format_graph_context(expanded)
        ans = generate_llm_answer(q, facts)
        answers.append(ans)

        # collect for response
        all_facts.append(f"Q: {q}\n{facts}")
        all_seeds_meta.extend([
            {"labels": list(n.labels), "name": n.get("name") or n.get("title"), "score": sc}
            for n, sc in cands
        ])

    # Reuse the answers we already generated
    final_answer = "\n\n".join(f"{i+1}. {a}" for i, a in enumerate(answers)) if len(answers) > 1 else answers[0]

    return {
        "answer": final_answer,
        "facts": "\n\n---\n\n".join(all_facts),
        "seeds": all_seeds_meta,
        "params": {
            "top_k": body.top_k,
            "hops": body.hops,
            "labels": labels,
            "use_hybrid": body.use_hybrid,
            "alpha_vec": body.alpha_vec,
            "beta_kw": body.beta_kw,
            "use_mmr": body.use_mmr,
            "use_cross_doc": body.use_cross_doc,
            "decompose": body.decompose if body.decompose is not None else "AUTO",
        }
    }

@app.post("/debug-search")
def debug_search(body: dict = Body(...)):
    """Debug endpoint to test search components separately"""
    try:
        question = body.get("question", "test")
        print(f"Debug search for: {question}")
        
        # Test vector search
        qvec = get_question_embedding(question)
        vector_results = vector_find_similar_nodes(qvec, DEFAULT_LABELS, top_k=5)
        
        # Test keyword search  
        keyword_results = fulltext_search(question, limit=5, labels=DEFAULT_LABELS)
        
        # Test hybrid
        hybrid_results = hybrid_candidates(question, qvec, DEFAULT_LABELS, k_vec=5, k_kw=5)
        
        # Check what labels exist
        with driver.session() as s:
            labels_result = s.run("CALL db.labels() YIELD label RETURN collect(label) as labels").single()
            existing_labels = labels_result["labels"] if labels_result else []
        
        # Check what indexes exist
        with driver.session() as s:
            indexes_result = s.run("""
                SHOW INDEXES 
                YIELD name, labelsOrTypes, type, properties 
                RETURN collect({name: name, labels: labelsOrTypes, type: type, properties: properties}) as indexes
            """).single()
            existing_indexes = indexes_result["indexes"] if indexes_result else []
        
        # Check if nodes exist with default labels
        label_counts = {}
        with driver.session() as s:
            for label in DEFAULT_LABELS:
                count_result = s.run(f"MATCH (n:{label}) RETURN count(n) as count").single()
                label_counts[label] = count_result["count"] if count_result else 0
        
        return {
            "question": question,
            "existing_labels": existing_labels,
            "existing_indexes": existing_indexes,
            "label_counts": label_counts,
            "vector_results_count": len(vector_results),
            "keyword_results_count": len(keyword_results),
            "hybrid_results_count": len(hybrid_results),
            "default_labels": DEFAULT_LABELS,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Debug search error: {e}")
        return {
            "error": str(e),
            "status": "error"
        }
    
@app.post("/ingest")
async def upload_and_ingest(file: UploadFile):
    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    task = ingest_markdown_task.delay(save_path)
    return {"job_id": task.id, "status": "queued"}

@app.get("/ingest/status/{job_id}")
def check_status(job_id: str):
    from celery_app import celery
    result = celery.AsyncResult(job_id)
    return {
        "job_id": job_id,
        "state": result.state,
        "result": result.result
    }