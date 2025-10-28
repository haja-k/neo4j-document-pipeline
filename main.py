import time
import re, os, uuid
from typing import List, Optional
from fastapi import FastAPI, Body, UploadFile, Request
from neo4j.exceptions import ServiceUnavailable
from fastapi.middleware.cors import CORSMiddleware
from tasks import ingest_markdown_task
from pydantic import BaseModel, Field
from time import perf_counter
from neo4j_connect import driver
from graphutil import (
    init_clients,
    get_question_embedding,
    hybrid_candidates,
    mmr_select,
    diversify_by_document,
    traverse_neighbors,
    format_graph_context,
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

# NEW: ensure embedding/chat clients are initialized at app startup
@app.on_event("startup")
async def _startup():
    await init_clients()  # if your init is sync, change to: init_clients()

class RagBody(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = 10
    hops: int = 1
    labels: Optional[List[str]] = None
    alpha_vec: float = 0.6
    beta_kw: float = 0.4
    use_mmr: bool = True
    use_cross_doc: bool = True

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

@app.post("/drop-all")
def drop_all_nodes(confirmation: bool = Body(False)):
    """Drop all nodes and relationships from the Neo4j database.
    
    Args:
        confirmation (bool): Must be set to true to confirm deletion
        
    Returns:
        dict: Status of the operation including node count before and after
    """
    if not confirmation:
        return {
            "status": "error",
            "message": "Confirmation required. Set confirmation=true in request body to proceed with deletion."
        }
        
    try:
        with driver.session() as session:
            # Get initial count
            initial_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            
            # Delete all relationships and nodes
            session.run("MATCH (n) DETACH DELETE n")
            
            # Verify deletion
            final_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            
            return {
                "status": "success",
                "message": "Successfully cleared the database",
                "nodes_before": initial_count,
                "nodes_after": final_count
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to clear database: {str(e)}"
        }
    
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
async def graphrag(body: RagBody = Body(...), request: Request = None):  # async + Request
    if not body.question.strip():
        return {"answer": "Please provide a question.", "facts": "", "seeds": []}

    timings = {}
    t_total = perf_counter()

    q0 = body.question.strip()
    labels = body.labels or DEFAULT_LABELS

    # 1) Embed question (AWAIT)
    t = perf_counter()
    qvec = await get_question_embedding(q0, request=request)
    timings["embed"] = perf_counter() - t

    # 2) Hybrid candidates (vector + keyword)
    t = perf_counter()
    cands = hybrid_candidates(
        question=q0, qvec=qvec, labels=labels,
        k_vec=max(12, body.top_k), k_kw=max(12, body.top_k),
        alpha_vec=body.alpha_vec, beta_kw=body.beta_kw
    )
    timings["hybrid"] = perf_counter() - t

    # 3) MMR diversification
    if body.use_mmr and len(cands) > body.top_k:
        t = perf_counter()
        cands = mmr_select(cands, k=body.top_k)
        timings["mmr"] = perf_counter() - t
    else:
        cands = cands[:body.top_k]

    # 4) Cross-document coverage
    if body.use_cross_doc and len(cands) > 1:
        t = perf_counter()
        cands = diversify_by_document(cands, k=len(cands))
        timings["cross_doc"] = perf_counter() - t

    # 5) Expand neighbors
    t = perf_counter()
    seed_ids = [n.element_id for n, _ in cands]
    expanded = traverse_neighbors(
        seed_ids,
        max_hops=max(1, min(body.hops, 3))
    )
    timings["graph_traverse"] = perf_counter() - t

    # 6) Format context (final step now)
    t = perf_counter()
    facts = format_graph_context(expanded, max_lines=None, snippet_chars=None, include_source=True)
    timings["format_context"] = perf_counter() - t
    
    # No LLM step — return the context as the "answer"
    ans = facts

    # seeds meta
    seeds_meta = [
        {"labels": list(n.labels), "name": n.get("name") or n.get("title"), "score": sc}
        for n, sc in cands
    ]

    timings["total"] = perf_counter() - t_total
    print(f"[TIMINGS] {timings}")

    return {
        "answer": ans,
        "facts": f"Q: {q0}\n{facts}",
        "seeds": seeds_meta,
        "params": {
            "top_k": body.top_k,
            "hops": body.hops,
            "labels": labels,
            "alpha_vec": body.alpha_vec,
            "beta_kw": body.beta_kw,
            "use_mmr": body.use_mmr,
            "use_cross_doc": body.use_cross_doc,
            "used_llm": False,
        },
        "timings": timings,
    }
    
@app.post("/debug-search")
def debug_search(body: dict = Body(...)):
    """Debug endpoint to test search components separately"""
    try:
        question = body.get("question", "test")
        print(f"Debug search for: {question}")
        
        # Test vector search
        qvec = get_question_embedding(question)

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
    dir_path = os.path.join(UPLOAD_DIR, file_id)
    os.makedirs(dir_path, exist_ok=True)
    save_path = os.path.join(dir_path, file.filename)
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