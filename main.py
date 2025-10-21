import re, os, uuid
from typing import List, Optional
from fastapi import FastAPI, Body, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tasks import ingest_markdown_task
from pydantic import BaseModel, Field

from neo4j_connect import driver
from graphutil import (
    get_question_embedding,
    hybrid_candidates,
    mmr_select,
    diversify_by_document,
    traverse_neighbors,
    format_graph_context,
    generate_llm_answer,
    decompose_question,
    DEFAULT_LABELS,
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

@app.post("/graphrag")
def graphrag(body: RagBody = Body(...)):
    if not body.question.strip():
        return {"answer": "Please provide a question.", "facts": "", "seeds": []}

    q0 = body.question.strip()

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

        # 2) Hybrid candidates (vector + keyword)
        cands = hybrid_candidates(
            question=q, qvec=qvec, labels=labels,
            k_vec=max(12, body.top_k), k_kw=max(12, body.top_k),
            alpha_vec=body.alpha_vec, beta_kw=body.beta_kw
        )

        # 3) MMR diversification
        if body.use_mmr and len(cands) > body.top_k:
            cands = mmr_select(cands, k=body.top_k)
        else:
            cands = cands[:body.top_k]

        # 4) Cross-document coverage
        if body.use_cross_doc and len(cands) > 1:
            cands = diversify_by_document(cands, k=len(cands))

        # 5) Expand neighbors (no need to open a session just to read element ids)
        seed_ids = [n.element_id for n, _ in cands]
        expanded = traverse_neighbors(
            seed_ids,
            max_hops=max(1, min(body.hops, 3))
        )

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