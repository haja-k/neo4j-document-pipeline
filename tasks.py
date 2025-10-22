from celery_app import celery
from ingestMD import process_file  # assuming this is your ingestion function

@celery.task(name="tasks.ingest_markdown_task", bind=True)
def ingest_markdown_task(self, file_path: str):
    try:
        # You can optionally send progress updates using self.update_state()
        process_file(file_path, in_memory_chunk_cache={})
        return {"status": "completed", "file": file_path}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

# Register this task when imported
celery.tasks.register(ingest_markdown_task)
