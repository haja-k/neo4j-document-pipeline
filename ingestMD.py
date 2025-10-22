import os
import re
import json
import yaml
import hashlib
import tiktoken
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable
from neo4j_connect import driver
from openai import AzureOpenAI

# -----------------------------
# Load config
# -----------------------------
with open("config/embedConfig.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

openai_cfg = cfg["azure_configs"]
client = AzureOpenAI(
    api_key=cfg["api_key"],
    api_version=openai_cfg["api_version"],
    azure_endpoint=openai_cfg["base_url"],
)

EMBED_MODEL = openai_cfg["embedding_deployment"]
CHAT_MODEL  = openai_cfg["chat_deployment"]

SAFE = re.compile(r"[^A-Za-z0-9_]")
ENC = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE    = int(cfg.get("chunking", {}).get("size", 1500))
CHUNK_OVERLAP = int(cfg.get("chunking", {}).get("overlap", 200))
EMBED_BATCH_SIZE = int(cfg.get("chunking", {}).get("embed_batch", 256))

# -----------------------------
# Preflight: Azure deployments
# -----------------------------
def assert_azure_deployments():
    err_hint = (
        "\nHint: In Azure OpenAI Studio, create deployments and set their **deployment names** "
        "into embedConfig.yaml under azure_configs.embedding_deployment & chat_deployment."
    )
    try:
        client.embeddings.create(input=["hi"], model=EMBED_MODEL)
    except Exception as e:
        raise RuntimeError(f"Embedding deployment '{EMBED_MODEL}' invalid: {e}{err_hint}")
    try:
        client.chat.completions.create(model=CHAT_MODEL, messages=[{"role":"user","content":"ping"}], temperature=0)
    except Exception as e:
        raise RuntimeError(f"Chat deployment '{CHAT_MODEL}' invalid: {e}{err_hint}")

# ---------------------------------
# Utilities: hashing & normalization
# ---------------------------------
def safe_label(x: str, fallback: str = "Entity") -> str:
    """Sanitize labels / rel types for Cypher."""
    x = (x or fallback).strip().replace(" ", "_")
    return SAFE.sub("_", x)[:64]

def canonicalize_text_keep_paragraphs(text: str) -> str:
    """
    Normalize while preserving paragraph boundaries.
    - Normalize line endings to \n
    - Collapse 3+ newlines to exactly 2 (paragraph break)
    - Trim trailing spaces
    - Collapse long runs of spaces/tabs on the same line
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)                   # keep paragraph breaks
    t = re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE)  # strip line-end spaces
    t = re.sub(r"[ \t]{2,}", " ", t)                   # compress inline spaces
    return t.strip()

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def batch_list(seq: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """Yield fixed-size batches from seq."""
    batch = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

# -----------------------------
# Chunking (robust, token-based)
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
    """
    Two-stage chunking:
    1) Pack paragraphs into chunks up to max_tokens
    2) Any leftover chunk > max_tokens will be token-sliced
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    tcount = 0

    for p in paras:
        p_tokens = ENC.encode(p)
        if len(p_tokens) > max_tokens:
            # flush buffer first
            if buf:
                chunks.append("\n\n".join(buf))
                buf, tcount = [], 0
            # split this very long paragraph
            chunks.extend(_split_by_tokens(p, max_tokens, overlap_tokens))
            continue

        # +1 token slack for the join/newline
        if tcount + len(p_tokens) + 1 > max_tokens and buf:
            chunks.append("\n\n".join(buf))
            buf, tcount = [p], len(p_tokens)
        else:
            buf.append(p)
            tcount += len(p_tokens) + 1

    if buf:
        chunks.append("\n\n".join(buf))

    # Second pass safety: ensure all chunks obey the cap
    final_chunks: List[str] = []
    for c in chunks:
        if len(ENC.encode(c)) <= max_tokens:
            final_chunks.append(c)
        else:
            final_chunks.extend(_split_by_tokens(c, max_tokens, overlap_tokens))

    # Debug: quick peek at first few chunk sizes
    print(f"Total chunks: {len(final_chunks)}")
    for i, c in enumerate(final_chunks[:5], 1):
        print(f"  • chunk[{i}] chars={len(c):,} tokens≈{len(ENC.encode(c)):,}")
    return final_chunks

# --------------------
# LLM: triple extract
# --------------------
def extract_triples(text_chunk: str) -> List[Dict[str, Any]]:
    system_msg = (
        "You are a graph ontology extractor. From the given policy text, extract structured triples as JSON. "
        "Each triple must contain: subject, predicate, object, subject_type, object_type. "
        "Subject and object types should be one of: Goal, Strategy, Challenge, Outcome, Policy, Stakeholder, Sector, Pillar, "
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

        print("🔎 Cleaned JSON output preview:", content[:200].replace("\n", " "))
        triples = json.loads(content)

        if not isinstance(triples, list):
            return []
        triples = [t for t in triples if isinstance(t, dict)]
        print(f"Extracted {len(triples)} triples")
        return triples
    except Exception as e:
        msg = str(e)
        if "DeploymentNotFound" in msg:
            print("Failed to extract triples: Azure 404 DeploymentNotFound – check your deployment names in embedConfig.yaml")
        else:
            print("Failed to extract triples:", e)
        return []

# ----------------------------
# Embeddings (batched + cache)
# ----------------------------
_emb_cache: Dict[str, List[float]] = {}  # text -> vector

def collect_unique_texts(triples: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    """Return lists of unique subject-texts, object-texts, relation-texts to embed."""
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

def embed_batch(texts: List[str]) -> None:
    """Embed a list of texts, skipping ones already in cache. Fills _emb_cache."""
    to_do = [t for t in texts if t not in _emb_cache]
    if not to_do:
        return
    for batch in batch_list(to_do, EMBED_BATCH_SIZE):
        try:
            resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
        except Exception as e:
            print("Embedding batch error:", e)
            # Best effort: fall back to singletons for this batch to isolate bad input
            for t in batch:
                try:
                    r = client.embeddings.create(input=[t], model=EMBED_MODEL)
                    _emb_cache[t] = r.data[0].embedding
                except Exception as ee:
                    print("  └─ Failed on item:", t[:80], "…", ee)
            continue
        for t, d in zip(batch, resp.data):
            _emb_cache[t] = d.embedding

def get_embeddings_for_chunk(triples: List[Dict[str, Any]]) -> None:
    """Collect unique texts and fill _emb_cache via batched calls."""
    subs, objs, rels = collect_unique_texts(triples)
    all_texts = subs + objs + rels
    embed_batch(all_texts)

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

    # 1) Ensure embeddings are ready (batched)
    get_embeddings_for_chunk(triples)

    # 2) Upsert Document (outer merge ensures doc exists even if rows don't write title/path)
    with driver.session() as session:
        session.run(
            """
            MERGE (d:Document {doc_id:$doc_id})
              ON CREATE SET d.title=$title, d.path=$path, d.createdAt=timestamp()
              ON MATCH  SET d.title=coalesce(d.title,$title), d.path=coalesce(d.path,$path), d.updatedAt=timestamp()
            """,
            {"doc_id": doc_id, "title": document_name, "path": full_path},
        )

        # 3) Prepare rows grouped by (labels, rel)
        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for t in triples:
            s, o, p = t.get("subject"), t.get("object"), t.get("predicate")
            if not (s and o and p):
                continue
            sub_type = safe_label(t.get("subject_type"), "Entity")
            obj_type = safe_label(t.get("object_type"), "Entity")
            rel_type = safe_label(p, "RELATED_TO")

            sub_key = f"{sub_type}:{s}"
            obj_key = f"{obj_type}:{o}"
            rel_key = f"{s} {p} {o}"

            sub_emb = _emb_cache.get(sub_key)
            obj_emb = _emb_cache.get(obj_key)
            rel_emb = _emb_cache.get(rel_key)
            if not (sub_emb and obj_emb and rel_emb):
                print("  ! Missing embedding; skipping triple:", (s, p, o))
                continue

            k = (sub_type, obj_type, rel_type)
            groups.setdefault(k, []).append({
                "s": s,
                "o": o,
                "doc_id": doc_id,
                "doc_title": document_name,   # <-- include for safety
                "doc_path": full_path,        # <-- include for safety
                "sub_emb": sub_emb,
                "obj_emb": obj_emb,
                "rel_emb": rel_emb,
                "src_txt": (source_text or "")[:1000],
            })

        inserted_total = 0

        # 4) Execute one UNWIND per (sub_type, obj_type, rel_type) group
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
                rel.sources     = CASE
                                    WHEN r.doc_id IS NULL THEN []
                                    ELSE [r.doc_id]
                                  END,
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
                d.path      = coalesce(r.doc_path,  d.path),
                d.createdAt = timestamp()
            ON MATCH  SET
                d.title     = coalesce(d.title, r.doc_title),
                d.path      = coalesce(d.path,  r.doc_path),
                d.updatedAt = timestamp()

            // Provenance: Document -> Node (MENTIONS)
            MERGE (d)-[ms:MENTIONS]->(sub)
            ON CREATE SET ms.createdAt = timestamp()

            MERGE (d)-[mo:MENTIONS]->(obj)
            ON CREATE SET mo.createdAt = timestamp()

            // Reverse provenance for browsing from nodes: Node -> Document (SOURCE)
            MERGE (sub)-[:SOURCE]->(d)
            MERGE (obj)-[:SOURCE]->(d)

            // Optional: carry a short snippet on the relation
            WITH rel, r.src_txt AS chunkTxt
            FOREACH (_ IN CASE WHEN chunkTxt IS NOT NULL AND size(chunkTxt) > 0 THEN [1] ELSE [] END |
              SET rel.source_text = chunkTxt
            )

            RETURN count(*) AS wrote
            """
            try:
                res = session.run(cypher, {"rows": rows}).single()
                wrote = res["wrote"] if res else 0
                inserted_total += wrote
            except Exception as e:
                print(f" ❌ Failed batch ({sub_type})-[{rel_type}]->({obj_type}):", e)

        print(f"✅ Stored/updated {inserted_total} triples for document: {document_name} ({doc_id[:8]}…)")

def create_doc_constraints():
    with driver.session() as session:
        session.run("""
        CREATE CONSTRAINT doc_docid_unique IF NOT EXISTS
        FOR (d:Document) REQUIRE d.doc_id IS UNIQUE
        """)

def detect_embedding_dimensions() -> int:
    """Detect the actual embedding dimensions from Azure OpenAI"""
    try:
        test_text = "test embedding dimensions"
        response = client.embeddings.create(input=[test_text], model=EMBED_MODEL)
        dimensions = len(response.data[0].embedding)
        print(f"🔍 Detected embedding dimensions: {dimensions}")
        return dimensions
    except Exception as e:
        print(f"❌ Could not detect embedding dimensions: {e}")
        return 1536  # Safe fallback

def create_vector_indexes():
    """Create vector indexes with correct dimensions - DROP AND RECREATE"""
    dimensions = detect_embedding_dimensions()
    
    labels = [
        "Document", "Stakeholder", "Goal", "Challenge", "Outcome", "Policy",
        "Strategy", "Pillar", "Sector", "Time_Period", "Infrastructure",
        "Technology", "Initiative", "Objective", "Target", "Opportunity",
        "Vision", "Region", "Enabler", "Entity",
    ]
    
    with driver.session() as session:
        # First, drop ALL existing vector indexes to avoid conflicts
        print("🗑️  Dropping existing vector indexes...")
        for label in labels:
            index_name = f"{label}_embedding_index"
            try:
                session.run(f"DROP INDEX {index_name} IF EXISTS")
            except Exception as e:
                print(f"⚠️  Could not drop {index_name}: {e}")
        
        # Then create all indexes with correct dimensions
        print("🧠 Creating new vector indexes...")
        for label in labels:
            index_name = f"{label}_embedding_index"
            try:
                session.run(f"""
                CREATE VECTOR INDEX {index_name}
                FOR (n:{label}) ON (n.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """)
                print(f"✅ Created vector index: {index_name} ({dimensions}D)")
            except Exception as e:
                print(f"❌ Failed to create vector index {index_name}: {e}")

def create_fulltext_index():
    """Create proper fulltext index"""
    with driver.session() as session:
        try:
            # Drop existing fulltext index
            session.run("DROP INDEX node_text_fulltext IF EXISTS")
            
            # Create new fulltext index with correct labels
            session.run("""
            CREATE FULLTEXT INDEX node_text_fulltext
            FOR (n:Document|Stakeholder|Goal|Challenge|Outcome|Pillar|Entity|Policy|Strategy|Sector) 
            ON EACH [n.name, n.title]
            """)
            print("✅ Created fulltext index: node_text_fulltext")
        except Exception as e:
            print(f"❌ Failed to create fulltext index: {e}")

def create_name_indexes():
    """Optional but recommended: speed up MERGE by (name)."""
    labels = [
        "Stakeholder","Goal","Challenge","Outcome","Policy","Strategy","Pillar","Sector",
        "Time_Period","Infrastructure","Technology","Initiative","Objective","Target",
        "Opportunity","Vision","Region","Enabler","Entity"
    ]
    with driver.session() as session:
        for label in labels:
            idx = f"{label}_name_idx"
            print(f"🔎 Creating name index: {idx}")
            session.run(f"CREATE INDEX {idx} IF NOT EXISTS FOR (n:{label}) ON (n.name)")

# --------------- NEW: auto-discover Markdown ---------------
def discover_markdown(base_dir: Optional[Path] = None, recursive: bool = False) -> List[str]:
    """
    Return Markdown file paths (as strings) from the current directory.
    Set recursive=True to include subfolders.
    """
    base = base_dir or Path.cwd()
    patterns = ("*.md", "*.markdown", "*.mdx")
    picker = base.rglob if recursive else base.glob
    files: set[Path] = set()
    for pat in patterns:
        files.update(p for p in picker(pat) if p.is_file())
    # Sort by name for consistent processing order
    return [str(p.resolve()) for p in sorted(files, key=lambda x: x.name.lower())]

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
            print(f" 🔁 Chunk {i}: cache hit ({len(triples)} triples)")
        else:
            print(f" 🔎 Chunk {i}: extracting triples…")
            triples = extract_triples(chunk)
            in_memory_chunk_cache[chunk_id] = triples

        if triples:
            store_in_neo4j(
                triples,
                document_name=filename,
                doc_id=doc_id,
                full_path=abs_path,
                source_text=chunk[:600],
            )
            total_triples += len(triples)
        else:
            print("  …No triples extracted for this chunk")

    print(f"\n✅ Done: {filename} | Total triples: {total_triples}")

if __name__ == "__main__":
    assert_azure_deployments()
    
    # CREATE INDEXES FIRST
    print("🔧 Creating indexes...")
    create_doc_constraints()
    create_name_indexes()
    create_vector_indexes(dim=1536)  # Explicitly set to 1536
    create_fulltext_index()
    print("✅ All indexes created/verified")
    
    # Then process files
    files = discover_markdown(recursive=False)
    if not files:
        print(f"⚠️  No Markdown files found under {Path.cwd()}")
        raise SystemExit(0)

    chunk_cache: Dict[str, List[Dict[str, Any]]] = {}
    for fp in files:
        process_file(fp, in_memory_chunk_cache=chunk_cache)

    print("\n✅ Ingestion complete.")


# python ingestMD.py
### Note: Just put the markdownfile wished to be ingested in the same directory