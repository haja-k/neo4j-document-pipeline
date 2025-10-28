from celery_app import celery
from ingestMD import (
    process_file,
    assert_azure_chat,
    show_embedding_banner,
    create_name_indexes,
    create_doc_constraints,
    create_vector_indexes,
)

@celery.task(name="tasks.ingest_markdown_task", bind=True)
def ingest_markdown_task(self, file_path: str):
    try:
        # --- preflight (same as CLI) ---
        assert_azure_chat()          # CLI calls this first :contentReference[oaicite:3]{index=3}
        show_embedding_banner()      # CLI prints model & dimension :contentReference[oaicite:4]{index=4}

        # --- ingest the uploaded file (CLI loops files; API does one) ---
        chunk_cache = {}
        process_file(file_path, in_memory_chunk_cache=chunk_cache)  # :contentReference[oaicite:5]{index=5}

        # --- post: ensure indexes/constraints (CLI does this) ---
        create_name_indexes()        # name indexes per label :contentReference[oaicite:6]{index=6}
        create_doc_constraints()     # Document.doc_id UNIQUE :contentReference[oaicite:7]{index=7}
        create_vector_indexes()      # vector indexes per label; auto-detect dim :contentReference[oaicite:8]{index=8}

        return {"status": "completed", "file": file_path}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
