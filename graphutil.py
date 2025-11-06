# graphutil.py
import os
import math
import re
import yaml
import httpx
import asyncio
import time
from typing import List, Dict, Any, Iterable, Tuple, Optional
from neo4j_connect import driver
from openai import AzureOpenAI

# ===============================
# Persistent Clients & Config
# ===============================
_cfg = None
_openai = None
client = None
_vllm_http: Optional[httpx.AsyncClient] = None
CHAT_MODEL = None

async def init_clients():
    """Initialize persistent clients once."""
    global _cfg, _openai, client, _vllm_http, CHAT_MODEL

    if _cfg is None:
        with open("config/embedConfig.yaml", "r") as f:
            _cfg = yaml.safe_load(f)
        _openai = _cfg["azure_configs"]

    if client is None:
        client = AzureOpenAI(
            api_key=_cfg["api_key"],
            api_version=_openai["api_version"],
            azure_endpoint=_openai["base_url"]
        )

    if _vllm_http is None:
        api_key = os.getenv("SAINS_VLLM_API_KEY", _cfg.get("sains_vllm", {}).get("api_key", ""))
        _vllm_http = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(30.0, connect=10.0),  # Increased from 10s, separate connect timeout
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)  # Connection pooling
        )

    CHAT_MODEL = _openai["chat_deployment"]
    return _cfg, client, _vllm_http


async def close_clients(vllm_http):
    try:
        await vllm_http.aclose()
    except Exception:
        pass


# ===============================
# Provider Config
# ===============================
def _lazy_cfg():
    global _cfg, _openai
    if _cfg is None:
        with open("config/embedConfig.yaml", "r") as f:
            _cfg = yaml.safe_load(f)
        _openai = _cfg["azure_configs"]
    return _cfg, _openai


def _get_providers():
    _cfg, _openai = _lazy_cfg()
    provider = os.getenv("EMBEDDING_PROVIDER", _cfg.get("embedding_provider", "azure")).lower()
    return provider, _cfg, _openai


# ===============================
# Embedding Functions
# ===============================
def _azure_embed(text: str, max_retries: int = 3) -> List[float]:
    """Sync embedding call to Azure OpenAI with retry logic."""
    print("[DEBUG] >>> Using Azure sync embedding path")
    
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(input=[text], model=_openai["embedding_deployment"])
            return resp.data[0].embedding
        except Exception as e:
            error_name = type(e).__name__
            if attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                print(f"[AZURE-EMBED-RETRY] Attempt {attempt + 1}/{max_retries} failed with {error_name}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[AZURE-EMBED-ERROR] All {max_retries} attempts failed: {error_name}: {e}")
                raise


async def _sains_vllm_embed(text: str, request=None, max_retries: int = 3) -> List[float]:
    """Async embedding call to vLLM server with retry logic."""
    print("[DEBUG] >>> Using vLLM async embedding path")
    _, _cfg, _ = _get_providers()
    sains_cfg = _cfg.get("sains_vllm", {})
    base_url = sains_cfg.get("base_url", "").rstrip("/")
    model = sains_cfg.get("model", "Qwen/Qwen3-Embedding-8B")

    session = request.app.state.vllm_http if request and hasattr(request.app.state, "vllm_http") else _vllm_http

    payload = {"model": model, "input": [text]}

    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            resp = await session.post(f"{base_url}/embeddings", json=payload)
            t1 = time.perf_counter()
            print(f"[DEBUG] <<< vLLM call took {t1 - t0:.3f}s")

            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            error_name = type(e).__name__
            if attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt)
                print(f"[VLLM-EMBED-RETRY] Attempt {attempt + 1}/{max_retries} failed with {error_name}: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"[VLLM-EMBED-ERROR] All {max_retries} attempts failed: {error_name}: {e}")
                raise


async def get_question_embedding(question: str, request=None, max_retries: int = 3) -> List[float]:
    """Return embedding with retry logic for connection errors."""
    provider, *_ = _get_providers()
    print(provider)
    
    for attempt in range(max_retries):
        try:
            if provider == "sains_vllm":
                print("Hi")
                return await _sains_vllm_embed(question, request=request)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _azure_embed, question)
        except Exception as e:
            error_name = type(e).__name__
            if attempt < max_retries - 1:
                wait_time = 0.5 * (2 ** attempt)  # Exponential backoff: 0.5s, 1s, 2s
                print(f"[EMBED-RETRY] Attempt {attempt + 1}/{max_retries} failed with {error_name}: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"[EMBED-ERROR] All {max_retries} attempts failed: {error_name}: {e}")
                raise  # Re-raise after all retries exhausted


# ===============================
# Shared Constants
# ===============================
DEFAULT_LABELS = [
    "Stakeholder", "Goal", "Challenge", "Outcome", "Policy", "Strategy", "Pillar", "Sector",
    "Time_Period", "Infrastructure", "Technology", "Initiative", "Objective", "Target",
    "Opportunity", "Vision", "Region", "Enabler", "Entity"
]


# This variable is initialized by init_clients()
CHAT_MODEL = None


# ===============================
# Utility functions
# ===============================
def _cosine(a: List[float], b: List[float]) -> float:
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b):
        num += x * y
        da += x * x
        db += y * y
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (math.sqrt(da) * math.sqrt(db))


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

# =========================
# Keyword / Full-text search
# =========================
FTS_NAME = "node_text_fulltext"
_LUCENE_ESC_RE = re.compile(r'([+\-!(){}[\]^"~*?:\\/])')
_BOOL_RE = re.compile(r'\b(AND|OR|NOT)\b', flags=re.IGNORECASE)

def _sanitize_term_for_lucene(term: str) -> str:
    """
    Escape Lucene special characters and neutralize accidental boolean operators
    *inside a single term*. This keeps our external ' OR ' joiner functional.
    """
    s = _LUCENE_ESC_RE.sub(r'\\\1', term)
    s = _BOOL_RE.sub(r'\\\1', s)
    return s

def _ensure_fulltext_index(session, labels: List[str]):
    """Create a full-text index if it doesn't exist, over name/title across labels."""
    label_union = "|".join(f"`{l}`" for l in labels)
    session.run(f"""
    CREATE FULLTEXT INDEX `{FTS_NAME}` IF NOT EXISTS
    FOR (n:{label_union}) ON EACH [n.name, n.title]
    """)

def _extract_keywords(question: str, max_terms: int = 8) -> List[str]:
    """Ask GPT to extract keyword queries (phrases ok)."""
    sys = ("Extract up to {k} short keywords or phrases from the user's question. "
           "Return them as a JSON array of strings, no extra text.").format(k=max_terms)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role":"system","content":sys},
            {"role":"user","content":question}
        ],
        temperature=0
    )
    txt = resp.choices[0].message.content.strip()
    if txt.startswith("```"):
        txt = txt.split("```", 1)[-1]
    if txt.endswith("```"):
        txt = txt[:-3]
    import json
    try:
        arr = json.loads(txt)
        return [str(x) for x in arr if isinstance(x, (str,int,float))][:max_terms]
    except Exception:
        return question.split()[:max_terms]

_ANCHOR_RE = re.compile(r'"([^"]+)"|“([^”]+)”|‘([^’]+)’|\'([^\']+)\'')

def _anchor_terms(question: str, max_terms: int = 3) -> list[str]:
    """
    Pull out strongly indicative terms to force into full-text search.
    Priority:
      1) Quoted phrases e.g. "Route location"
      2) Title-cased bigrams e.g. Route Location
      3) Fallback to the longest keyword from _extract_keywords
    """
    anchors: list[str] = []

    # 1) quoted phrases
    for g in _ANCHOR_RE.findall(question):
        val = next((x for x in g if x), "").strip()
        if val and val.lower() not in ("and", "or", "the"):
            anchors.append(val)

    # 2) Title-cased bigrams
    if len(anchors) < max_terms:
        words = re.findall(r"[A-Za-z][A-Za-z\-]+", question)
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1[0].isupper() and w2[0].isupper():
                pair = f"{w1} {w2}"
                if pair not in anchors:
                    anchors.append(pair)
                    if len(anchors) >= max_terms:
                        break

    # 3) Fallback to longest keyword
    if not anchors:
        kws = _extract_keywords(question, max_terms=5)
        kws = sorted(kws, key=len, reverse=True)
        if kws:
            anchors.append(kws[0])

    # de-dupe, keep order
    seen = set()
    out = []
    for a in anchors:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out[:max_terms]

def _fulltext_query_string(terms: List[str]) -> str:
    """
    Build a Lucene-safe OR-joined query. Each *term* is sanitized before
    being optionally quoted (for multi-word phrases).
    """
    toks = []
    for t in terms:
        t = str(t).strip()
        if not t:
            continue
        t_safe = _sanitize_term_for_lucene(t)
        if " " in t:
            toks.append(f'"{t_safe}"')
        else:
            toks.append(t_safe)
    return " OR ".join(toks) if toks else ""

def fulltext_search(question: str, limit: int = 12, labels: Optional[List[str]] = None) -> List[Tuple[Any, float]]:
    """BM25 full-text search using Neo4j full-text index, anchored on key phrases."""
    labels = labels or DEFAULT_LABELS
    try:
        with driver.session() as s:
            _ensure_fulltext_index(s, labels)
            anchors = _anchor_terms(question, max_terms=3)
            kws     = _extract_keywords(question, max_terms=8)
            terms: List[str] = []
            seen = set()
            for t in anchors + kws:
                t = (t or "").strip()
                if not t:
                    continue
                if t.lower() in seen:
                    continue
                terms.append(t)
                seen.add(t.lower())

            q = _fulltext_query_string(terms)
            if not q:
                return []

            lim = max(limit, 16)

            res = s.run(
                """
                CALL db.index.fulltext.queryNodes($name, $q, {limit: $lim})
                YIELD node, score
                RETURN node, score
                """,
                name=FTS_NAME, q=q, lim=lim
            )
            hits = [(r["node"], float(r["score"])) for r in res]

            if not hits and anchors:
                aq = _fulltext_query_string(anchors[:1])
                if aq:
                    res2 = s.run(
                        """
                        CALL db.index.fulltext.queryNodes($name, $q, {limit: $lim})
                        YIELD node, score
                        RETURN node, score
                        """,
                        name=FTS_NAME, q=aq, lim=lim
                    )
                    hits = [(r["node"], float(r["score"])) for r in res2]

            return hits[:limit]
    except Exception as e:
        print(f"ERROR in fulltext_search: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []

# =========================
# Vector search (base)
# =========================
def _query_label_index(session, label: str, top_k: int, qvec: List[float]) -> List[Tuple[Any, float]]:
    """Search a single label's vector index; return [(node, score)]"""
    index_name = f"{label}_embedding_index"
    res = session.run(
        """
        CALL db.index.vector.queryNodes($index, $k, $vec)
        YIELD node, score
        RETURN node, score
        """,
        index=index_name, k=top_k, vec=qvec
    )
    return [(r["node"], float(r["score"])) for r in res]

def vector_find_similar_nodes(qvec: List[float], labels: Iterable[str], top_k: int = 12) -> List[Tuple[Any, float]]:
    """Query across multiple label indexes and merge deduped results."""
    labels = list(labels) if labels else DEFAULT_LABELS
    results: Dict[str, Tuple[Any, float]] = {}
    with driver.session() as s:
        for label in labels:
            try:
                for node, score in _query_label_index(s, label, top_k, qvec):
                    eid = node.element_id
                    if eid not in results or score > results[eid][1]:
                        results[eid] = (node, score)
            except Exception as e:
                print(f"[vector] skip {label}: {e}")
    merged = sorted(results.values(), key=lambda t: t[1], reverse=True)
    return merged[:top_k]

# =========================
# Hybrid (vector + keyword) — confidence removed
# =========================
def _node_embedding(node) -> Optional[List[float]]:
    """Return node.embedding as a Python list (or None)."""
    e = node.get("embedding")
    return list(e) if e is not None else None

def hybrid_candidates(question: str,
                      qvec: List[float],
                      labels: Optional[List[str]] = None,
                      k_vec: int = 12,
                      k_kw: int = 12,
                      alpha_vec: float = 0.6,
                      beta_kw: float = 0.25) -> List[Tuple[Any, float]]:
    """
    Hybrid retrieval: combine vector and fulltext scores (no confidence term).
    Returns list of (node, combined_score).
    """
    labels = labels or DEFAULT_LABELS
    vec_hits = vector_find_similar_nodes(qvec, labels, top_k=k_vec)
    kw_hits  = fulltext_search(question, limit=k_kw, labels=labels)

    # collect raw
    raw: Dict[str, Dict[str, Any]] = {}
    # vector
    for n, sc in vec_hits:
        eid = n.element_id
        raw.setdefault(eid, {"node": n, "vec": 0.0, "kw": 0.0})
        raw[eid]["vec"] = max(raw[eid]["vec"], float(sc))
    # keyword
    for n, sc in kw_hits:
        eid = n.element_id
        raw.setdefault(eid, {"node": n, "vec": 0.0, "kw": 0.0})
        raw[eid]["kw"] = max(raw[eid]["kw"], float(sc))

    # normalize per channel
    vec_norm = _minmax_norm([v["vec"] for v in raw.values()])
    kw_norm  = _minmax_norm([v["kw"] for v in raw.values()])
    for (entry, vn, kn) in zip(raw.values(), vec_norm, kw_norm):
        entry["vec_n"] = vn
        entry["kw_n"]  = kn

    # blend vector + keyword only (renormalize weights to sum to 1)
    w_sum = max(1e-12, (alpha_vec + beta_kw))
    w_vec = alpha_vec / w_sum
    w_kw  = beta_kw  / w_sum

    out: List[Tuple[Any, float]] = []
    for entry in raw.values():
        combined = w_vec*entry["vec_n"] + w_kw*entry["kw_n"]
        out.append((entry["node"], combined))

    out.sort(key=lambda t: t[1], reverse=True)
    return out

# =========================
# MMR diversification
# =========================
def mmr_select(candidates: List[Tuple[Any, float]],
               k: int,
               lambda_mult: float = 0.7) -> List[Tuple[Any, float]]:
    """
    Maximal Marginal Relevance over candidate nodes.
    Needs node.embeddings; falls back to score-only if embeddings missing.
    """
    if not candidates:
        return []

    # fetch embeddings
    with driver.session() as s:
        embs: List[Optional[List[float]]] = []
        for n, _ in candidates:
            embs.append(_node_embedding(n))

    selected: List[int] = []
    rest = list(range(len(candidates)))
    # seed: best score
    best0 = max(rest, key=lambda i: candidates[i][1])
    selected.append(best0); rest.remove(best0)

    def _max_sim_to_selected(j: int) -> float:
        ej = embs[j]
        if ej is None or not selected:
            return 0.0
        sims = []
        for i in selected:
            ei = embs[i]
            if ei is None:
                sims.append(0.0)
            else:
                sims.append(_cosine(ej, ei))
        return max(sims) if sims else 0.0

    while len(selected) < min(k, len(candidates)):
        best_j, best_val = None, -1e9
        for j in rest:
            relevance = candidates[j][1]
            diversity_penalty = _max_sim_to_selected(j)
            val = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if val > best_val:
                best_val, best_j = val, j
        selected.append(best_j); rest.remove(best_j)

    return [candidates[i] for i in selected]

# =========================
# Cross-document coverage
# =========================
def _doc_title_for_node(session, node):
    rec = session.run("""
      MATCH (n) WHERE elementId(n) = $id
      OPTIONAL MATCH (n)-[:SOURCE]->(d1:Document)
      OPTIONAL MATCH (d2:Document)-[:MENTIONS]->(n)
      RETURN coalesce(d1.title, d2.title) AS t
      LIMIT 1
    """, id=node.element_id).single()
    return rec["t"] if rec and rec["t"] is not None else None

def diversify_by_document(cands: List[Tuple[Any, float]], k: int) -> List[Tuple[Any, float]]:
    """
    Prefer seeds from different documents (round-robin by doc title).
    """
    if not cands:
        return []
    buckets: Dict[str, List[Tuple[Any, float]]] = {}
    with driver.session() as s:
        for n, sc in cands:
            t = _doc_title_for_node(s, n) or "__NO_DOC__"
            buckets.setdefault(t, []).append((n, sc))
    # sort buckets by best score inside
    for b in buckets.values():
        b.sort(key=lambda t: t[1], reverse=True)
    order = sorted(buckets.keys(), key=lambda key: buckets[key][0][1], reverse=True)

    picked: List[Tuple[Any, float]] = []
    ptrs = {k:0 for k in buckets.keys()}
    while len(picked) < min(k, len(cands)):
        progressed = False
        for key in order:
            i = ptrs[key]
            if i < len(buckets[key]):
                picked.append(buckets[key][i])
                ptrs[key] += 1
                progressed = True
                if len(picked) >= k:
                    break
        if not progressed:
            break
    return picked

# =========================
# Expansion, formatting, answering
# =========================
def traverse_neighbors(seed_element_ids: List[str],
                       max_hops: int = 1) -> Dict[str, Any]:
    """
    Expand neighborhood around seeds. Uses APOC if present; otherwise pure Cypher.
    Neo4j 5 compatible.
    """
    if not seed_element_ids:
        return {"nodes": [], "rels": []}
    
    try:
        with driver.session() as s:
            try:
                apoc_ok = s.run("""
                    SHOW PROCEDURES YIELD name
                    WHERE name = 'apoc.path.expandConfig'
                    RETURN count(*) AS c
                """).single()["c"] > 0
            except Exception:
                apoc_ok = False

            if apoc_ok:
                q = """
                MATCH (n)
                WHERE elementId(n) IN $ids
                CALL apoc.path.expandConfig(n, {
                  minLevel: 1,
                  maxLevel: $hops,
                  uniqueness: "NODE_GLOBAL",
                  bfs: true
                }) YIELD path
                WITH collect(path) AS paths
                UNWIND paths AS p
                WITH collect(nodes(p)) AS nlists, collect(relationships(p)) AS rlists
                UNWIND nlists AS nl
                UNWIND nl AS n
                WITH collect(DISTINCT n) AS ns, rlists
                UNWIND rlists AS rl
                UNWIND rl AS r
                WITH ns, collect(DISTINCT r) AS rs2
                RETURN
                  [x IN ns | {props: properties(x), labels: labels(x), id: elementId(x)}] AS nodes,
                  [r IN rs2 | {
                      type: type(r),
                      props: properties(r),
                      start: elementId(startNode(r)),
                      end: elementId(endNode(r))
                  }] AS rels
                """
                rec = s.run(q, ids=seed_element_ids, hops=max_hops).single()
                if rec is None:
                    return {"nodes": [], "rels": []}
                return {"nodes": rec["nodes"] or [], "rels": rec["rels"] or []}

            # ---------- Pure Cypher fallback (no APOC at all) ----------
            q = """
            MATCH (n)
            WHERE elementId(n) IN $ids
            MATCH p=(n)-[*1..$hops]-(m)
            WITH collect(p) AS paths
            UNWIND paths AS p
            WITH collect(nodes(p)) AS nlists, collect(relationships(p)) AS rlists
            UNWIND nlists AS nl
            UNWIND nl AS n
            WITH collect(DISTINCT n) AS ns, rlists
            UNWIND rlists AS rl
            UNWIND rl AS r
            WITH ns, collect(DISTINCT r) AS rs2
            RETURN
              [x IN ns | {props: properties(x), labels: labels(x), id: elementId(x)}] AS nodes,
              [r IN rs2 | {
                  type: type(r),
                  props: properties(r),
                  start: elementId(startNode(r)),
                  end: elementId(endNode(r))
              }] AS rels
            """
            rec = s.run(q, ids=seed_element_ids, hops=max_hops).single()
            if rec is None:
                return {"nodes": [], "rels": []}
            return {"nodes": rec["nodes"] or [], "rels": rec["rels"] or []}
    except Exception as e:
        print(f"ERROR in traverse_neighbors: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Return empty result instead of crashing
        return {"nodes": [], "rels": []}

# --------- Dedup helpers ---------
_NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

def _normalize_name_for_key(name: str) -> str:
    """
    Normalize node names for dedup keys:
    - lowercase
    - strip punctuation
    - collapse spaces
    - map small number words -> digits (e.g., 'six' -> '6')
    """
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = " ".join(_NUM_WORDS.get(tok, tok) for tok in s.split())
    return s

def _dedup_rels_by_key(rels: List[Dict[str, Any]], nodes_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop duplicate relations where (start_name, type, end_name) match after normalization.
    Keeps the first occurrence (preserving its original props/snippet).
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rels:
        s = nodes_by_id.get(r.get("start"), {}) or {}
        t = nodes_by_id.get(r.get("end"), {}) or {}
        s_props = (s.get("props") or {})
        t_props = (t.get("props") or {})
        s_name = s_props.get("name") or s_props.get("title") or ""
        t_name = t_props.get("name") or t_props.get("title") or ""
        rtype  = (r.get("type") or "").lower()

        key = (_normalize_name_for_key(s_name), rtype, _normalize_name_for_key(t_name))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def _doc_titles_for_nodes(element_ids: List[str]) -> Dict[str, Optional[str]]:
    """
    Return {elementId(node) -> document title or None}.
    Looks for either (node)-[:SOURCE]->(d:Document) or (d:Document)-[:MENTIONS]->(node).
    Batched for speed.
    """
    if not element_ids:
        return {}
    ids = list({i for i in element_ids if i})
    with driver.session() as s:
        recs = s.run("""
        UNWIND $ids AS id
        MATCH (n) WHERE elementId(n) = id
        OPTIONAL MATCH (n)-[:SOURCE]->(d1:Document)
        OPTIONAL MATCH (d2:Document)-[:MENTIONS]->(n)
        RETURN id, coalesce(d1.title, d2.title) AS title
        """, ids=ids)
        out: Dict[str, Optional[str]] = {}
        for r in recs:
            out[r["id"]] = r["title"]
        return out

def format_graph_context(
    expanded: dict,
    max_lines: int | None = None,
    snippet_chars: int | None = None,
    include_source: bool = False,
) -> str:
    """
    Build the 'Graph Facts' text.
    - No clipping when `max_lines is None` and `snippet_chars is None`.
    - Still skips noisy SOURCE/MENTIONS edges.
    - When include_source=True, appends [source: "Document Title"] (or both if different).
    """
    if not expanded:
        return "Graph Facts: (no results)"

    nodes = {n.get("id"): n for n in expanded.get("nodes", [])}
    rels = expanded.get("rels", []) or []
    rels = [r for r in rels if r.get("type") not in {"SOURCE", "MENTIONS"}]
    rels = _dedup_rels_by_key(rels, nodes)
    
    if not rels:
        return "Graph Facts: (no results)"

    titles: Dict[str, Optional[str]] = {}
    if include_source and rels:
        id_pool: List[str] = []
        for r in rels:
            if r.get("start"): id_pool.append(r["start"])
            if r.get("end"):   id_pool.append(r["end"])
        titles = _doc_titles_for_nodes(id_pool)

    # decide how many to show
    rels_iter = rels[:max_lines] if isinstance(max_lines, int) and max_lines > 0 else rels

    lines = ["Graph Facts:"]
    for r in rels_iter:
        s = nodes.get(r.get("start"), {}) or {}
        t = nodes.get(r.get("end"), {}) or {}

        s_props = s.get("props") or {}
        t_props = t.get("props") or {}
        s_name = s_props.get("name") or s_props.get("title") or "?"
        t_name = t_props.get("name") or t_props.get("title") or "?"
        s_label = (s.get("labels") or ["Entity"])[0]
        t_label = (t.get("labels") or ["Entity"])[0]

        rprops = r.get("props") or {}
        raw = (rprops.get("source_text_full") or rprops.get("source_text") or "").replace("\n", " ").strip()

        # unlimited snippet unless a positive `snippet_chars` is provided
        if isinstance(snippet_chars, int) and snippet_chars > 0 and len(raw) > snippet_chars:
            snip = raw[:snippet_chars].rstrip() + "..."
        else:
            snip = raw

        snip_str = f' [snippet: "{snip}"]' if snip else ""
        src_str = ""
        if include_source:
            ts = titles.get(r.get("start"))
            te = titles.get(r.get("end"))
            if ts and te and ts != te:
                src_str = f' [source: "{ts}" | "{te}"]'
            elif ts or te:
                src_str = f' [source: "{ts or te}"]'

        lines.append(
            f'- {s_label}("{s_name}") -[{r.get("type")}]-> {t_label}("{t_name}"){snip_str}{src_str}'
        )

    return "\n".join(lines)

