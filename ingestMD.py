import os
import re
import json
import yaml
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable
from neo4j_connect import driver
from openai import AzureOpenAI
import httpx
import tiktoken

# -----------------------------
# Load config
# -----------------------------
with open("config/embedConfig.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# --- Azure (chat only; used for triple extraction) ---
openai_cfg = cfg["azure_configs"]
client = AzureOpenAI(
    api_key=cfg["api_key"],
    api_version=openai_cfg["api_version"],
    azure_endpoint=openai_cfg["base_url"],
)
CHAT_MODEL = openai_cfg["chat_deployment"]
sains_cfg   = cfg.get("sains_vllm", {})  # expects base_url, api_key, model
VLLM_BASE   = sains_cfg.get("base_url", "").rstrip("/")
VLLM_KEY    = sains_cfg.get("api_key", "")
VLLM_MODEL  = sains_cfg.get("model", "Qwen/Qwen3-Embedding-8B")
SAFE = re.compile(r"[^A-Za-z0-9_]")
ENC = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE = int(cfg.get("chunking", {}).get("size", 1500))
CHUNK_OVERLAP = 0 
_EMBED_DIM: Optional[int] = None

# -----------------------------
# Preflight: Azure (chat only)
# -----------------------------
def assert_azure_chat():
    """Validate the Azure chat deployment (we use it for triple extraction only)."""
    try:
        client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
        )
    except Exception as e:
        raise RuntimeError(
            f"Chat deployment '{CHAT_MODEL}' invalid: {e}\n"
            "Hint: set azure_configs.chat_deployment in embedConfig.yaml."
        )

# ---------------------------------
# Utilities: hashing & normalization
# ---------------------------------
def safe_label(x: str, fallback: str = "Entity") -> str:
    x = (x or fallback).strip().replace(" ", "_")
    return SAFE.sub("_", x)[:64]

def canonicalize_text_keep_paragraphs(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -----------------------------
# Chunking
# -----------------------------
def _split_by_tokens(text: str, size: int, overlap: int) -> List[str]:
    ids = ENC.encode(text)
    out = []
    step = max(1, size - overlap)
    for start in range(0, len(ids), step):
        piece_ids = ids[start:start+size]
        out.append(ENC.decode(piece_ids))
    return out

def chunk_markdown(text: str, max_tokens: int = CHUNK_SIZE, overlap_tokens: int = CHUNK_OVERLAP) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    tcount = 0

    for p in paras:
        p_tokens = ENC.encode(p)
        if len(p_tokens) > max_tokens:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, tcount = [], 0
            chunks.extend(_split_by_tokens(p, max_tokens, overlap_tokens))
            continue

        if tcount + len(p_tokens) + 1 > max_tokens and buf:
            chunks.append("\n\n".join(buf))
            buf, tcount = [p], len(p_tokens)
        else:
            buf.append(p)
            tcount += len(p_tokens) + 1

    if buf:
        chunks.append("\n\n".join(buf))

    final_chunks: List[str] = []
    for c in chunks:
        if len(ENC.encode(c)) <= max_tokens:
            final_chunks.append(c)
        else:
            final_chunks.extend(_split_by_tokens(c, max_tokens, overlap_tokens))

    print(f"Total chunks: {len(final_chunks)}")
    for i, c in enumerate(final_chunks[:5], 1):
        print(f"  • chunk[{i}] chars={len(c):,} tokens≈{len(ENC.encode(c)):,}")
    return final_chunks

# --------------------
# LLM: triple extract (Azure chat)
# --------------------
def extract_triples(text_chunk: str) -> List[Dict[str, Any]]:
    system_msg = (
        "You are a graph ontology extractor. From the given policy text, extract structured triples as JSON. "
        "Each triple must contain: subject, predicate, object, subject_type, object_type. "
        "Subject and object types should be one of: Goal, Strategy, Challenge, Outcome, Policy, Stakeholder, Sector, Pillar, Infrastructure, Technology, Initiative, Objective, Target, Opportunity, Time_Period, Vision "
        "or use 'Entity' if none fits. Output only a valid JSON list."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Text:\n{text_chunk}"},
    ]
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("```", 1)[-1]
        if content.startswith("json"):
            content = content[4:]
        if content.endswith("```"):
            content = content[:-3]
        start, end = content.find("["), content.rfind("]")
        if start != -1 and end != -1:
            content = content[start:end+1]
        triples = json.loads(content)
        if not isinstance(triples, list):
            return []
        return [t for t in triples if isinstance(t, dict)]
    except Exception as e:
        print("Failed to extract triples:", e)
        return []

# ----------------------------
# Fallback: extract triples from Markdown tables
# ----------------------------
def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _is_table_sep(line: str) -> bool:
    # detects a markdown header separator like: | --- | :---: | ---: |
    return bool(re.match(r'^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$', line))

def _split_cells(line: str) -> List[str]:
    # split a markdown row into cells; keep empty cells as ""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]

def extract_triples_from_markdown_tables(text: str) -> List[Dict[str, Any]]:
    """
    Convert markdown tables into triples:
      - First column = subject
      - Other columns = predicate -> object
      - subject_type='Entity', object_type='Value'
    """
    lines = text.splitlines()
    triples: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" in line:
            j = i + 1
            if j < len(lines) and _is_table_sep(lines[j]):
                headers = [_norm_space(h) for h in _split_cells(line)]
                k = j + 1
                # consume data rows until a blank or a non-table-ish line
                while k < len(lines) and "|" in lines[k] and not lines[k].strip().startswith(">"):
                    row = _split_cells(lines[k])
                    if len(row) >= 2 and row[0].strip():
                        subj = _norm_space(row[0])
                        for col in range(1, min(len(row), len(headers))):
                            pred = _norm_space(headers[col])
                            obj  = _norm_space(row[col])
                            if pred and obj:
                                triples.append({
                                    "subject": subj,
                                    "predicate": pred,
                                    "object": obj,
                                    "subject_type": "Entity",
                                    "object_type": "Value",
                                })
                    else:
                        break
                    k += 1
                i = k
                continue
        i += 1
    return triples


# ----------------------------
# Embeddings via vLLM/Qwen
# ----------------------------
_vllm_http: Optional[httpx.Client] = None
_emb_cache: Dict[str, List[float]] = {}  

def _get_vllm_client() -> httpx.Client:
    global _vllm_http
    if _vllm_http is None:
        if not (VLLM_BASE and VLLM_KEY):
            raise RuntimeError("Check embedConfig.yaml")
        _vllm_http = httpx.Client(
            base_url=VLLM_BASE,
            headers={"Authorization": f"Bearer {VLLM_KEY}"},
            timeout=60.0,
        )
    return _vllm_http

def _embed_with_vllm(texts: List[str]) -> List[List[float]]:
    http = _get_vllm_client()
    r = http.post("/embeddings", json={"model": VLLM_MODEL, "input": texts})
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data", [])
    data_sorted = sorted(data, key=lambda d: d.get("index", 0))
    embs = [d["embedding"] for d in data_sorted]
    if not embs or len(embs) != len(texts):
        raise RuntimeError(f"Unexpected embeddings response size: got {len(embs)}, expected {len(texts)}")
    global _EMBED_DIM
    if _EMBED_DIM is None:
        _EMBED_DIM = len(embs[0])
    return embs

def collect_unique_texts(triples: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    sub_texts = set()
    obj_texts = set()
    rel_texts = set()
    for t in triples:
        s, o, p = t.get("subject"), t.get("object"), t.get("predicate")
        if not (s and o and p):
            continue
        sub_type = safe_label(t.get("subject_type"), "Entity")
        obj_type = safe_label(t.get("object_type"), "Entity")
        sub_texts.add(f"{sub_type}:{s}")
        obj_texts.add(f"{obj_type}:{o}")
        rel_texts.add(f"{s} {p} {o}")
    return list(sub_texts), list(obj_texts), list(rel_texts)

def get_embeddings_for_chunk(triples: List[Dict[str, Any]]) -> None:
    subs, objs, rels = collect_unique_texts(triples)
    all_texts = subs + objs + rels
    to_do = [t for t in all_texts if t not in _emb_cache]
    if not to_do:
        return
    try:
        embs = _embed_with_vllm(to_do)
        for t, e in zip(to_do, embs):
            _emb_cache[t] = e
    except Exception as e:
        print("Embedding error via vLLM/Qwen:", e)
        # as a last-resort, try one by one (still no batching param)
        for t in to_do:
            try:
                _emb_cache[t] = _embed_with_vllm([t])[0]
            except Exception as ee:
                print("  └─ failed on item:", t[:80], "…", ee)

def _to_json_compact(x: Any) -> str:
    try:
        return json.dumps(x, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(x)

def _ensure_text_embeddings(texts: List[str]) -> None:
    """Ensure embeddings for arbitrary text keys (uses global _emb_cache)."""
    to_do = [t for t in texts if t not in _emb_cache]
    if not to_do:
        return
    try:
        embs = _embed_with_vllm(to_do)
        for t, e in zip(to_do, embs):
            _emb_cache[t] = e
    except Exception as e:
        print("Embedding error (special rows):", e)
        # last resort: one-by-one
        for t in to_do:
            try:
                _emb_cache[t] = _embed_with_vllm([t])[0]
            except Exception as ee:
                print("  └─ failed on special text:", t[:80], "…", ee)

def _is_timeline_obj(o: Any) -> bool:
    return isinstance(o, dict) and {"start", "end"} <= set(o.keys())

def _is_distribution_obj(o: Any) -> bool:
    # outer dict of {pillar: {term: count, ...}, ...}
    return isinstance(o, dict) and any(isinstance(v, dict) for v in o.values())

def _clean_name(s: Any) -> Optional[str]:
    if s is None:
        return None
    if isinstance(s, (str, int, float, bool)):
        return str(s)
    return _to_json_compact(s)

# ----------------------------------------
# Neo4j storage with batched writes
# ----------------------------------------
def store_in_neo4j(
    triples: List[Dict[str, Any]],
    document_name: str,
    doc_id: str,
    full_path: str,
    source_text: Optional[str] = None,
):
    if not triples:
        return

    # Keep your original embedding call for "normal" triples.
    # (It will embed dicts as their string repr; that's fine, we will override
    #  with special-row embeddings below where needed.)
    get_embeddings_for_chunk(triples)

    with driver.session() as session:
        # Ensure Document node exists
        session.run(
            """
            MERGE (d:Document {doc_id:$doc_id})
              ON CREATE SET d.title=$title, d.path=$path, d.createdAt=timestamp()
              ON MATCH  SET d.title=coalesce(d.title,$title), d.path=coalesce(d.path,$path), d.updatedAt=timestamp()
            """,
            {"doc_id": doc_id, "title": document_name, "path": full_path},
        )

        # --- Buckets ---
        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}  # normal SPO
        rows_timeline: List[Dict[str, Any]] = []                        # special timeline rows
        rows_dist: List[Dict[str, Any]] = []                            # special distribution rows

        # --- Partition triples into normal / special ---
        for t in triples:
            s, o, p = t.get("subject"), t.get("object"), t.get("predicate")
            if not (s and p):
                continue

            sub_type = safe_label(t.get("subject_type"), "Entity")
            obj_type = safe_label(t.get("object_type"), "Entity")
            rel_type = safe_label(p, "RELATED_TO")

            # Timeline object: {start, end}
            if _is_timeline_obj(o):
                start = str(o.get("start", "")).strip()
                end   = str(o.get("end", "")).strip()
                if not (start and end):
                    # if malformed timeline, fall back to normal with stringified object
                    o_name = _clean_name(o)
                    if not o_name:
                        continue
                    sub_key = f"{sub_type}:{s}"
                    obj_key = f"{obj_type}:{o_name}"
                    rel_key = f"{s} {p} {o_name}"
                    sub_emb = _emb_cache.get(sub_key)
                    obj_emb = _emb_cache.get(obj_key)
                    rel_emb = _emb_cache.get(rel_key)
                    if not (sub_emb and obj_emb and rel_emb):
                        # ensure (rare) missing embeddings
                        _ensure_text_embeddings([sub_key, obj_key, rel_key])
                        sub_emb = _emb_cache.get(sub_key)
                        obj_emb = _emb_cache.get(obj_key)
                        rel_emb = _emb_cache.get(rel_key)
                    groups.setdefault((sub_type, obj_type, rel_type), []).append({
                        "s": s, "o": o_name,
                        "doc_id": doc_id, "doc_title": document_name, "doc_path": full_path,
                        "sub_emb": sub_emb, "obj_emb": obj_emb, "rel_emb": rel_emb,
                        "src_txt": (source_text or "")[:1000],
                    })
                    continue

                # Prefer original predicate label if it's something like hasTimeline
                rel_label = safe_label(p, "HAS_TIMELINE")
                # Create a canonical name for the period to embed by
                tp_name = f"{start}–{end}"

                # Ensure embeddings for subject, time period (by name), and the rel text
                sub_key = f"{sub_type}:{s}"
                obj_key = f"Time_Period:{tp_name}"
                rel_key = f"{s} {p} {tp_name}"
                _ensure_text_embeddings([sub_key, obj_key, rel_key])

                rows_timeline.append({
                    "s": s,
                    "sub_type": sub_type,
                    "rel_label": rel_label,         # e.g., HAS_TIMELINE or HASTIMELINE
                    "tp_name": tp_name,
                    "start": start,
                    "end": end,
                    "doc_id": doc_id, "doc_title": document_name, "doc_path": full_path,
                    "sub_emb": _emb_cache.get(sub_key),
                    "obj_emb": _emb_cache.get(obj_key),
                    "rel_emb": _emb_cache.get(rel_key),
                    "src_txt": (source_text or "")[:1000],
                })
                continue

            # Distribution object: { pillar -> { term -> count, ... }, ... }
            if _is_distribution_obj(o):
                # Use a consistent relationship type for counts
                rel_label = "HAS_INITIATIVE_COUNT"
                for pillar, term_map in o.items():
                    if not isinstance(term_map, dict):
                        continue
                    pillar_name = str(pillar).strip()
                    if not pillar_name:
                        continue
                    for term, cnt in term_map.items():
                        if str(term).lower() == "total":
                            continue
                        try:
                            cnt_val = int(cnt)
                        except Exception:
                            # allow numeric-ish strings
                            try:
                                cnt_val = int(float(str(cnt)))
                            except Exception:
                                continue

                        # Use Pillar as the target label
                        sub_key = f"{sub_type}:{s}"
                        obj_key = f"Pillar:{pillar_name}"
                        rel_key = f"{s} {rel_label} {pillar_name}"
                        _ensure_text_embeddings([sub_key, obj_key, rel_key])

                        rows_dist.append({
                            "s": s,
                            "sub_type": sub_type,
                            "pillar": pillar_name,
                            "term": str(term),
                            "count": cnt_val,
                            "rel_label": rel_label,
                            "doc_id": doc_id, "doc_title": document_name, "doc_path": full_path,
                            "sub_emb": _emb_cache.get(sub_key),
                            "obj_emb": _emb_cache.get(obj_key),
                            "rel_emb": _emb_cache.get(rel_key),
                            "src_txt": (source_text or "")[:1000],
                        })
                continue

            # --- Normal path (strings or other primitives) ---
            o_name = _clean_name(o)
            if not o_name:
                # nothing usable
                continue

            sub_key = f"{sub_type}:{s}"
            obj_key = f"{obj_type}:{o_name}"
            rel_key = f"{s} {p} {o_name}"

            sub_emb = _emb_cache.get(sub_key)
            obj_emb = _emb_cache.get(obj_key)
            rel_emb = _emb_cache.get(rel_key)
            if not (sub_emb and obj_emb and rel_emb):
                _ensure_text_embeddings([sub_key, obj_key, rel_key])
                sub_emb = _emb_cache.get(sub_key)
                obj_emb = _emb_cache.get(obj_key)
                rel_emb = _emb_cache.get(rel_key)

            groups.setdefault((sub_type, obj_type, rel_type), []).append({
                "s": s,
                "o": o_name,
                "doc_id": doc_id,
                "doc_title": document_name,
                "doc_path": full_path,
                "sub_emb": sub_emb,
                "obj_emb": obj_emb,
                "rel_emb": rel_emb,
                "src_txt": (source_text or "")[:1000],
            })

        inserted_total = 0

        # 1) Normal triples
        for (sub_type, obj_type, rel_type), rows in groups.items():
            if not rows:
                continue
            cypher = f"""
            UNWIND $rows AS r
            MERGE (sub:`{sub_type}` {{name:r.s}})
            ON CREATE SET sub.embedding = r.sub_emb
            ON MATCH  SET sub.embedding = r.sub_emb

            MERGE (obj:`{obj_type}` {{name:r.o}})
            ON CREATE SET obj.embedding = r.obj_emb
            ON MATCH  SET obj.embedding = r.obj_emb

            MERGE (sub)-[rel:`{rel_type}`]->(obj)
            ON CREATE SET
                rel.embedding   = r.rel_emb,
                rel.sources     = CASE WHEN r.doc_id IS NULL THEN [] ELSE [r.doc_id] END,
                rel.createdAt   = timestamp()
            ON MATCH  SET
                rel.embedding   = r.rel_emb,
                rel.sources     = CASE
                                    WHEN r.doc_id IS NULL THEN rel.sources
                                    WHEN rel.sources IS NULL THEN [r.doc_id]
                                    WHEN NOT r.doc_id IN rel.sources THEN rel.sources + [r.doc_id]
                                    ELSE rel.sources
                                  END,
                rel.updatedAt   = timestamp()

            MERGE (d:Document {{doc_id: r.doc_id}})
            ON CREATE SET
                d.title     = coalesce(r.doc_title, d.title),
                d.path      = coalesce(r.doc_path,  r.doc_path),
                d.createdAt = timestamp()
            ON MATCH  SET
                d.title     = coalesce(d.title, r.doc_title),
                d.path      = coalesce(d.path,  r.doc_path),
                d.updatedAt = timestamp()

            MERGE (d)-[:MENTIONS]->(sub)
            MERGE (d)-[:MENTIONS]->(obj)
            MERGE (sub)-[:SOURCE]->(d)
            MERGE (obj)-[:SOURCE]->(d)

            WITH rel, r.src_txt AS chunkTxt
            FOREACH (_ IN CASE WHEN chunkTxt IS NOT NULL AND size(chunkTxt) > 0 THEN [1] ELSE [] END |
            SET rel.source_text       = substring(chunkTxt, 0, 1000),  // short preview
                rel.source_text_full = chunkTxt                        // full text
            )
            RETURN count(*) AS wrote
            """
            try:
                res = session.run(cypher, {"rows": rows}).single()
                wrote = res["wrote"] if res else 0
                inserted_total += wrote
            except Exception as e:
                print(f" ❌ failed batch ({sub_type})-[{rel_type}]->({obj_type}):", e)

        # 2) Timeline rows
        if rows_timeline:
            cypher_tl = """
            UNWIND $rows AS r
            WITH r,
                 r.sub_type AS sub_type
            CALL apoc.merge.node([sub_type], {name:r.s}) YIELD node AS sub
            SET sub.embedding = r.sub_emb

            MERGE (tp:Time_Period {start:r.start, end:r.end})
            ON CREATE SET tp.name = coalesce(r.tp_name, r.start + "–" + r.end),
                          tp.embedding = r.obj_emb
            ON MATCH  SET tp.embedding = r.obj_emb

            CALL apoc.create.relationship(sub, r.rel_label, {}, tp) YIELD rel
            SET rel.embedding = r.rel_emb,
                rel.sources   = CASE WHEN r.doc_id IS NULL THEN [] ELSE [r.doc_id] END,
                rel.createdAt = coalesce(rel.createdAt, timestamp()),
                rel.updatedAt = timestamp()

            MERGE (d:Document {doc_id: r.doc_id})
            ON CREATE SET d.title = coalesce(r.doc_title, d.title),
                          d.path  = coalesce(r.doc_path,  d.path),
                          d.createdAt = timestamp()
            ON MATCH  SET d.title = coalesce(d.title, r.doc_title),
                          d.path  = coalesce(d.path,  r.doc_path),
                          d.updatedAt = timestamp()

            MERGE (d)-[:MENTIONS]->(sub)
            MERGE (d)-[:MENTIONS]->(tp)
            MERGE (sub)-[:SOURCE]->(d)
            MERGE (tp)-[:SOURCE]->(d)
            RETURN count(*) AS wrote
            """
            try:
                res = session.run(cypher_tl, {"rows": rows_timeline}).single()
                wrote = res["wrote"] if res else 0
                inserted_total += wrote
            except Exception as e:
                print(" ❌ failed timeline batch:", e)

        # 3) Distribution rows
        if rows_dist:
            cypher_dist = """
            UNWIND $rows AS r
            WITH r,
                 r.sub_type AS sub_type
            CALL apoc.merge.node([sub_type], {name:r.s}) YIELD node AS sub
            SET sub.embedding = r.sub_emb

            MERGE (pl:Pillar {name:r.pillar})
            ON CREATE SET pl.embedding = r.obj_emb
            ON MATCH  SET pl.embedding = r.obj_emb

            CALL apoc.merge.relationship(sub, r.rel_label, {term:r.term}, {}, pl) YIELD rel
            SET rel.count     = r.count,
                rel.embedding = r.rel_emb,
                rel.sources   = CASE WHEN r.doc_id IS NULL THEN [] ELSE [r.doc_id] END,
                rel.createdAt = coalesce(rel.createdAt, timestamp()),
                rel.updatedAt = timestamp()

            MERGE (d:Document {doc_id: r.doc_id})
            ON CREATE SET d.title = coalesce(r.doc_title, d.title),
                          d.path  = coalesce(r.doc_path,  d.path),
                          d.createdAt = timestamp()
            ON MATCH  SET d.title = coalesce(d.title, r.doc_title),
                          d.path  = coalesce(d.path,  r.doc_path),
                          d.updatedAt = timestamp()

            MERGE (d)-[:MENTIONS]->(sub)
            MERGE (d)-[:MENTIONS]->(pl)
            MERGE (sub)-[:SOURCE]->(d)
            MERGE (pl)-[:SOURCE]->(d)
            RETURN count(*) AS wrote
            """
            try:
                res = session.run(cypher_dist, {"rows": rows_dist}).single()
                wrote = res["wrote"] if res else 0
                inserted_total += wrote
            except Exception as e:
                print(" ❌ failed distribution batch:", e)

        print(f"✅ stored/updated {inserted_total} triples for document: {document_name} ({doc_id[:8]}…)")

def create_doc_constraints():
    with driver.session() as session:
        session.run("""
        CREATE CONSTRAINT doc_docid_unique IF NOT EXISTS
        FOR (d:Document) REQUIRE d.doc_id IS UNIQUE
        """)

# --------------------------
# Vector + name indexes
# --------------------------
def _ensure_embed_dim() -> int:
    """
    Discover the embedding dimension without any defaults.
    We prefer a cached vector; otherwise request a tiny probe.
    """
    global _EMBED_DIM
    if _EMBED_DIM is not None:
        return _EMBED_DIM
    _EMBED_DIM = len(_embed_with_vllm(["dimension probe"])[0])
    return _EMBED_DIM

def create_vector_indexes():
    dim = _ensure_embed_dim()
    labels = [
        "Document", "Stakeholder", "Goal", "Challenge", "Outcome", "Policy",
        "Strategy", "Pillar", "Sector", "Time_Period", "Infrastructure",
        "Technology", "Initiative", "Objective", "Target", "Opportunity",
        "Vision", "Region", "Enabler", "Entity",
    ]
    with driver.session() as session:
        for label in labels:
            index_name = f"{label}_embedding_index"
            print(f"🧠 creating vector index: {index_name} (dim={dim})")
            session.run(f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON (n.embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
              }}
            }}
            """)

def create_name_indexes():
    labels = [
        "Stakeholder","Goal","Challenge","Outcome","Policy","Strategy","Pillar","Sector",
        "Time_Period","Infrastructure","Technology","Initiative","Objective","Target",
        "Opportunity","Vision","Region","Enabler","Entity"
    ]
    with driver.session() as session:
        for label in labels:
            idx = f"{label}_name_idx"
            print(f"🔎 creating name index: {idx}")
            session.run(f"CREATE INDEX {idx} IF NOT EXISTS FOR (n:{label}) ON (n.name)")

# --------------- auto-discover Markdown ---------------
def discover_markdown(base_dir: Optional[Path] = None, recursive: bool = False) -> List[str]:
    base = base_dir or Path.cwd()
    patterns = ("*.md", "*.markdown", "*.mdx")
    picker = base.rglob if recursive else base.glob
    files: set[Path] = set()
    for pat in patterns:
        files.update(p for p in picker(pat) if p.is_file())
    return [str(p.resolve()) for p in sorted(files, key=lambda x: x.name.lower())]

# ----------------
# Banner: show embedding model + dimension (exactly two lines)
# ----------------
def show_embedding_banner():
    dim_txt = "(probe failed)"
    try:
        dim = _ensure_embed_dim()
        dim_txt = str(dim)
    except Exception:
        pass
    print(f"Model: {VLLM_MODEL}")
    print(f"Detected embedding dimension: {dim_txt}")

# ----------------
# Main ingestion
# ----------------
def process_file(path: str, in_memory_chunk_cache: Dict[str, List[Dict[str, Any]]]):
    abs_path = str(Path(path).resolve())
    filename = Path(path).name

    with open(path, "r", encoding="utf-8") as f:
        content_raw = f.read()
    content_norm = canonicalize_text_keep_paragraphs(content_raw)
    doc_id = sha256_hex(content_norm)

    print(f"\n📄 Processing: {filename} ({doc_id[:8]}…) — re/ingesting regardless of prior state")

    chunks = chunk_markdown(content_norm, max_tokens=CHUNK_SIZE, overlap_tokens=CHUNK_OVERLAP)

    total_triples = 0
    for i, chunk in enumerate(chunks, 1):
        chunk_id = sha256_hex(chunk)
        if chunk_id in in_memory_chunk_cache:
            triples = in_memory_chunk_cache[chunk_id]
            print(f" 🔁 chunk {i}: cache hit ({len(triples)} triples)")
        else:
            print(f" 🔎 chunk {i}: extracting triples…")
            triples = extract_triples(chunk)
            in_memory_chunk_cache[chunk_id] = triples

        if triples:
            get_embeddings_for_chunk(triples)
            store_in_neo4j(
                triples,
                document_name=filename,
                doc_id=doc_id,
                full_path=abs_path,
                source_text=chunk[:1000],
            )
            total_triples += len(triples)
        else:
            table_triples = extract_triples_from_markdown_tables(chunk)
            if table_triples:
                print(f"  ↪︎ fallback tables: extracted {len(table_triples)} triples")
                get_embeddings_for_chunk(table_triples)
                store_in_neo4j(
                    table_triples,
                    document_name=filename,
                    doc_id=doc_id,
                    full_path=abs_path,
                    source_text=chunk[:1000],
                )
                total_triples += len(table_triples)
            else:
                print("  …no triples extracted for this chunk")

    print(f"\n✅ Done: {filename} | Total triples: {total_triples}")

if __name__ == "__main__":
    assert_azure_chat()
    show_embedding_banner()
    files = discover_markdown(recursive=False)
    if not files:
        print(f"⚠️  No Markdown files found under {Path.cwd()}")
        raise SystemExit(0)

    chunk_cache: Dict[str, List[Dict[str, Any]]] = {}
    for fp in files:
        process_file(fp, in_memory_chunk_cache=chunk_cache)

    create_name_indexes()
    create_doc_constraints()
    create_vector_indexes()
    print("\n✅ Ingestion and indexing complete.")
