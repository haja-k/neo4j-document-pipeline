# graphutil.py
import math
import re
import yaml
from typing import List, Dict, Any, Iterable, Tuple, Optional
from neo4j_connect import driver
from openai import AzureOpenAI

# --- Load config shared with ingest ---
with open("config/embedConfig.yaml", "r") as f:
    _cfg = yaml.safe_load(f)

_openai = _cfg["azure_configs"]
EMBED_MODEL = _openai["embedding_deployment"]
CHAT_MODEL  = _openai["chat_deployment"]

client = AzureOpenAI(
    api_key=_cfg["api_key"],
    api_version=_openai["api_version"],
    azure_endpoint=_openai["base_url"]
)

DEFAULT_LABELS = [
    "Stakeholder","Goal","Challenge","Outcome","Policy","Strategy","Pillar","Sector",
    "Time_Period","Infrastructure","Technology","Initiative","Objective","Target",
    "Opportunity","Vision","Region","Enabler","Entity"
]

# =========================
# Embeddings & utilities
# =========================
def get_question_embedding(question: str) -> List[float]:
    """Embed a user question for vector search."""
    emb = client.embeddings.create(input=[question], model=EMBED_MODEL)
    return emb.data[0].embedding

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

def build_retrieval_query(question: str, max_len: int = 240) -> str:
    """
    Produce a compact, anchor-first retrieval string that fits the tool's 256-char limit.
    Keeps quoted phrases / Title-Cased bigrams, then top keywords, trimmed to max_len.
    """
    anchors = _anchor_terms(question, max_terms=3)
    kws     = _extract_keywords(question, max_terms=8)

    terms, seen = [], set()
    for t in anchors + kws:
        t = (t or "").strip()
        if not t:
            continue
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        terms.append(f'"{t}"' if " " in t else t)

    out, total = [], 0
    for t in terms:
        add = (t + " ")
        if total + len(add) > max_len:
            break
        out.append(t)
        total += len(add)

    q = " ".join(out).strip()
    if not q:
        q = question[:max_len]
    return q

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


def get_labels_with_embeddings() -> List[str]:
    """Return only labels that actually have nodes with embeddings"""
    with driver.session() as s:
        result = s.run("""
        CALL db.labels() YIELD label
        MATCH (n:label) 
        WHERE n.embedding IS NOT NULL
        RETURN label, count(n) as count 
        ORDER BY count DESC
        """)
        return [record["label"] for record in result if record["count"] > 0]

def get_labels_with_embeddings() -> List[str]:
    """Return only labels that actually have nodes with embeddings"""
    with driver.session() as s:
        result = s.run("""
        CALL db.labels() YIELD label
        MATCH (n:label) 
        WHERE n.embedding IS NOT NULL
        RETURN label, count(n) as count 
        ORDER BY count DESC
        """)
        return [record["label"] for record in result if record["count"] > 0]

def vector_find_similar_nodes(qvec: List[float], labels: Iterable[str], top_k: int = 12) -> List[Tuple[Any, float]]:
    """Smart vector search that only queries labels with actual data"""
    labels = list(labels) if labels else DEFAULT_LABELS
    
    # Filter to only labels that have nodes with embeddings
    valid_labels = []
    with driver.session() as s:
        for label in labels:
            try:
                # Quick check if label has any nodes with embeddings
                result = s.run(
                    f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL RETURN count(n) as count LIMIT 1"
                ).single()
                if result and result["count"] > 0:
                    valid_labels.append(label)
                else:
                    print(f"[vector] skipping {label}: no nodes with embeddings")
            except Exception as e:
                print(f"[vector] label check failed for {label}: {e}")
    
    if not valid_labels:
        print("[vector] No valid labels found with embeddings")
        return []
    
    results: Dict[str, Tuple[Any, float]] = {}
    with driver.session() as s:
        for label in valid_labels:
            try:
                for node, score in _query_label_index(s, label, top_k, qvec):
                    eid = node.element_id
                    if eid not in results or score > results[eid][1]:
                        results[eid] = (node, score)
            except Exception as e:
                print(f"[vector] search failed for {label}: {e}")
                # Continue with other labels instead of failing completely
    
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
                       max_hops: int = 1,
                       min_confidence: float = 0.0) -> Dict[str, Any]:
    """
    Expand neighborhood around seeds. Uses APOC if present; otherwise pure Cypher.
    Neo4j 5 compatible. (min_confidence kept for API compatibility but unused)
    """
    if not seed_element_ids:
        return {"nodes": [], "rels": []}
        
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
            # FIX: Handle case where rec is None
            if rec is None:
                return {"nodes": [], "rels": []}
            return {"nodes": rec["nodes"], "rels": rec["rels"]}

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
        # FIX: Handle case where rec is None
        if rec is None:
            return {"nodes": [], "rels": []}
        return {"nodes": rec["nodes"], "rels": rec["rels"]}

def format_graph_context(expanded: dict, max_lines: int = 40) -> str:
    """
    Turn the expanded subgraph into a compact, LLM-friendly text context.
    - Skips noisy SOURCE and MENTIONS edges.
    - No confidence sorting or display.
    - Limits total lines via max_lines.
    """
    if not expanded or not expanded.get("nodes"):
        return "Graph Facts: (no results)"

    nodes = {n.get("id"): n for n in expanded.get("nodes", [])}
    rels = expanded.get("rels", []) or []
    rels = [r for r in rels if r.get("type") not in {"SOURCE", "MENTIONS"}]

    if not rels:
        return "Graph Facts: (no relationships found)"

    lines = ["Graph Facts:"]
    for r in rels[:max_lines]:
        s = nodes.get(r.get("start"), {})
        t = nodes.get(r.get("end"), {})

        s_props = s.get("props") or {}
        t_props = t.get("props") or {}
        s_name = s_props.get("name") or s_props.get("title") or "?"
        t_name = t_props.get("name") or t_props.get("title") or "?"
        s_label = (s.get("labels") or ["Entity"])[0]
        t_label = (t.get("labels") or ["Entity"])[0]

        rprops = r.get("props") or {}
        snip = (rprops.get("source_text") or "")[:200]
        snip_str = f' [snippet: "{snip}..."]' if snip else ""

        lines.append(
            f'- {s_label}("{s_name}") -[{r.get("type")}]-> {t_label}("{t_name}"){snip_str}'
        )

    return "\n".join(lines)

def generate_llm_answer(
    question: str,
    facts: str,
    max_tokens: int = 900,
    style: str = "detailed",
) -> str:
    """
    Answer ONLY from Graph Facts with maximal specificity:
      1) First extract every numeric target (%, RM, counts, MW), dates/years,
         and named projects/initiatives/entities that appear in the facts.
      2) Present a 'Quantitative Highlights' list with those items (no omissions).
      3) Then provide a structured narrative answer grouped by themes
         (Economic / Social / Environmental / Infrastructure/Technology / Governance/Policy).
      4) If something is missing in the facts, explicitly say: 'Not found in provided facts.'
      5) Never invent or speculate beyond the facts.
    """
    if style not in {"detailed", "concise"}:
        style = "detailed"

    section_hint = (
        "If the content spans multiple themes, group with these headers when relevant: "
        "Economic, Social, Environmental, Infrastructure/Technology, Governance/Policy."
    )

    extraction_instructions = (
        "Before drafting the answer, extract ALL quantitative items and proper nouns "
        "(numbers with units like %, RM, MW, jobs, #startups; dates/years; named projects/initiatives; "
        "named stakeholders and places) present in Graph Facts. "
        "List them under 'Quantitative Highlights' as bullet points. "
        "Do not include anything that is not explicitly present in the facts."
    )

    formatting_rules = (
        "Formatting rules:\n"
        "• Start with 'Quantitative Highlights' (every number/date/project found; no omissions).\n"
        "• Then give a one-sentence executive summary.\n"
        "• Then use short section headers and bullets (max ~12 bullets total unless asked for a long list).\n"
        "• Prefer exact figures and named entities from the facts; do NOT estimate.\n"
        "• If pillars/strategies are listed, include 1–2 concrete actions/initiatives under each IF present in facts.\n"
        "• If info is missing, write: 'Not found in provided facts.'\n"
        "• Do not cite or rely on outside knowledge.\n"
    )

    verbosity = (
        "Be comprehensive but crisp. Keep to ~150–220 words."
        if style == "concise"
        else "Be detailed but crisp. Keep to ~220–320 words."
    )

    system = (
        "You are a meticulous analyst. You must answer ONLY from the provided Graph Facts. "
        "Your first duty is to extract every quantitative detail and named project found in the facts. "
        "Never invent values beyond what is present. If missing, say so."
    )

    user = (
        f"Question:\n{question}\n\n"
        f"{section_hint}\n"
        f"{extraction_instructions}\n"
        f"{formatting_rules}\n"
        f"{verbosity}\n\n"
        f"Graph Facts (authoritative context):\n{facts}"
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Answer generation error: {e})"

# =========================
# Question decomposition
# =========================
def decompose_question(question: str, max_parts: int = 3) -> List[str]:
    """
    Use GPT to split complex questions into sub-questions (<= max_parts).
    """
    sys = (f"Split the user's question into up to {max_parts} minimal sub-questions, "
           "ordered logically. Return ONLY a JSON array of strings.")
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
        out = [str(x).strip() for x in arr if str(x).strip()]
        return out[:max_parts] if out else [question]
    except Exception:
        return [question]
