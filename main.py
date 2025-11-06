import time
import re, os, uuid
from typing import List, Optional
from fastapi import FastAPI, Body, UploadFile, Request
from neo4j.exceptions import ServiceUnavailable
from fastapi.middleware.cors import CORSMiddleware
from tasks import ingest_markdown_task
from pydantic import BaseModel, Field
from time import perf_counter
from neo4j_connect import driver, close_driver
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

@app.get("/documents")
def list_documents():
    """Get a list of all documents stored in Neo4j with their related stats.
    
    Returns:
        dict: Status and list of documents with their titles and related entity counts
    """
    try:
        with driver.session() as session:
            # Query documents with their entity counts
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:MENTIONS]->(e)
                WITH d, count(DISTINCT e) as entityCount
                OPTIONAL MATCH (n)-[:SOURCE]->(d)
                WITH d, entityCount, count(DISTINCT n) as sourceNodeCount
                RETURN {
                    title: d.title,
                    entityCount: entityCount,
                    sourceNodeCount: sourceNodeCount,
                    uploadedAt: d.uploadedAt
                } as document
                ORDER BY d.uploadedAt DESC
            """)
            documents = [record["document"] for record in result]
            
            return {
                "success": True,
                "message": "Documents retrieved successfully",
                "documents": documents
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to retrieve documents: {str(e)}"
        }

@app.get("/healthz")
def healthz():
    try:
        with driver.session() as s:
            c = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        return {"success": True, "nodes": c}
    except Exception as e:
        return {"success": False, "message": "Unhealthy. Error: " + str(e)}

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
            "success": False,
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
                "success": True,
                "message": "Successfully cleared the database",
                "nodes_before": initial_count,
                "nodes_after": final_count
            }
    except Exception as e:
        return {
            "success": False,
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

@app.on_event("shutdown")
def shutdown_event():
    """Close Neo4j driver on shutdown"""
    close_driver()
    print("✅ Neo4j driver closed")

@app.post("/graphrag")
async def graphrag(body: RagBody = Body(...), request: Request = None):  # async + Request
    try:
        if not body.question.strip():
            return {"success": False, "message": "Please provide a question.", "answer": "", "facts": "", "seeds": []}

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

        # 3) Early "no data" check
        if not cands:
            msg = "There is no available data related to the user query."
            return {
                "success": True,
                "message": msg,
                "answer": msg,
                "facts": f"Q: {q0}\nGraph Facts: (no results)",
                "seeds": [],
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

        # 4) MMR diversification
        if body.use_mmr and len(cands) > body.top_k:
            t = perf_counter()
            cands = mmr_select(cands, k=body.top_k)
            timings["mmr"] = perf_counter() - t
        else:
            cands = cands[:body.top_k]

        # 5) Cross-document coverage
        if body.use_cross_doc and len(cands) > 1:
            t = perf_counter()
            cands = diversify_by_document(cands, k=len(cands))
            timings["cross_doc"] = perf_counter() - t

        # 6) Expand neighbors
        t = perf_counter()
        raw_seed_ids = [n.element_id for n, _ in cands]
        with driver.session() as s:
            recs = s.run("""
                UNWIND $ids AS id
                MATCH (n) WHERE elementId(n) = id
                OPTIONAL MATCH (d:Document)-[:MENTIONS]->(n)
                WITH id, collect(d.doc_id) AS docs
                RETURN id, CASE WHEN size(docs) > 0 THEN docs[0] ELSE id END AS doc
            """, {"ids": raw_seed_ids}).data()

        seen_docs = set()
        seed_ids = []
        for r in recs:
            doc = r["doc"]
            node_id = r["id"]
            if doc in seen_docs:
                continue
            seen_docs.add(doc)
            seed_ids.append(node_id)
            
        expanded = traverse_neighbors(
            seed_ids,
            max_hops=max(1, min(body.hops, 3))
        )
        timings["graph_traverse"] = perf_counter() - t

        # 7) Format context
        t = perf_counter()
        facts = format_graph_context(expanded, max_lines=None, snippet_chars=None, include_source=True)
        timings["format_context"] = perf_counter() - t

        # If no facts at all, emit a clear, user-friendly message
        if facts.strip().endswith("(no results)"):
            msg = "There is no available data related to the user query."
            return {
                "success": True,
                "message": msg,
                "answer": msg,
                "facts": f"Q: {q0}\n{facts}",
                "seeds": [],
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
            "success": True,
            "message": "Query processed.",
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

    except ServiceUnavailable as e:
        # Log server-side
        print(f"Neo4j connection error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False, 
            "message": "Database temporarily unavailable. Please try again.",
            "error_type": "connection"
        }
    except Exception as e:
        # Log server-side
        print(f"graphrag error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Query failed. No knowledge able to be retrieved. Error: {str(e)}"}
    
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
            "success": True,
            "message": "Debug search completed",
            "question": question,
            "existing_labels": existing_labels,
            "existing_indexes": existing_indexes,
            "label_counts": label_counts,
            "hybrid_results_count": len(hybrid_results),
            "default_labels": DEFAULT_LABELS,
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Debug search error: {e}"
        }
    
@app.post("/ingest")
async def upload_and_ingest(file: UploadFile):
    """Upload a file and queue ingest task. Returns standardized response with success and message.

    Expects multipart/form-data with file field.
    """
    try:
        file_id = str(uuid.uuid4())
        dir_path = os.path.join(UPLOAD_DIR, file_id)
        os.makedirs(dir_path, exist_ok=True)
        save_path = os.path.join(dir_path, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        task = ingest_markdown_task.delay(save_path)
        return {"success": True, "message": "Ingestion queued.", "job_id": task.id}
    except Exception as e:
        return {"success": False, "message": f"Ingest error: {e}"}

@app.get("/queue_stats")
def get_queue_stats():
    """Get statistics about the Celery task queue including active, reserved, and scheduled tasks.
    
    Returns:
        dict: Queue statistics including counts of active, reserved, and scheduled tasks,
              and details about documents currently being processed
    """
    try:
        from celery_app import celery
        inspector = celery.control.inspect()
        
        # Get active tasks with their details
        active = inspector.active()
        active_tasks = []
        if active:
            for worker_tasks in active.values():
                for task in worker_tasks:
                    # Get document filename from task args if it's an ingest task
                    if task.get('name') == 'tasks.ingest_markdown_task':
                        try:
                            file_path = task.get('args', [None])[0]
                            if file_path:
                                filename = os.path.basename(file_path)
                                active_tasks.append({
                                    'id': task.get('id'),
                                    'filename': filename,
                                    'started_at': task.get('time_start'),
                                    'worker': task.get('worker')
                                })
                        except (IndexError, AttributeError):
                            pass
        active_count = len(active_tasks)
        
        # Get reserved tasks (tasks that have been claimed by workers but not yet started)
        reserved = inspector.reserved()
        reserved_count = sum(len(tasks) for tasks in (reserved or {}).values())
        
        # Get scheduled tasks
        scheduled = inspector.scheduled()
        scheduled_count = sum(len(tasks) for tasks in (scheduled or {}).values())
        
        # Get registered workers
        workers = inspector.ping() or {}
        worker_count = len(workers)
        
        # Get revoked tasks (cancelled or failed)
        revoked = inspector.revoked() or {}
        revoked_count = sum(len(tasks) for tasks in revoked.values())
        
        return {
            "success": True,
            "message": "Queue statistics retrieved successfully",
            "stats": {
                "active_tasks": active_count,
                "reserved_tasks": reserved_count,
                "scheduled_tasks": scheduled_count,
                "revoked_tasks": revoked_count,
                "total_in_progress": active_count + reserved_count + scheduled_count,
                "worker_count": worker_count,
                "workers": list(workers.keys()) if workers else [],
                "documents_in_progress": active_tasks
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to retrieve queue statistics: {str(e)}"
        }

@app.get("/ingest/status")
def check_status(job_id: str = None):
    """Check status of an ingest job
    
    Args:
        job_id (str): The job ID from the ingest task
    """
    if not job_id:
        return {
            "success": False,
            "message": "Missing value: job_id parameter is required."
        }
        
    from celery_app import celery
    result = celery.AsyncResult(job_id)
    return {
        "success": True,
        "message": "Job status retrieved",
        "job_id": job_id,
        "state": result.state,
        "result": result.result
    }

@app.get("/graph/stats")
def get_graph_statistics():
    """Get comprehensive statistics about the Neo4j graph database.
    
    Returns:
        dict: Various statistics about nodes, relationships, and database state
    """
    try:
        with driver.session() as session:
            # Get node statistics by label
            label_stats = session.run("""
                CALL db.labels() YIELD label
                CALL {
                    WITH label
                    MATCH (n:`${label}`)
                    RETURN count(n) as count
                }
                RETURN label, count
                ORDER BY label
            """)
            node_counts = {record["label"]: record["count"] for record in label_stats}
            
            # Get relationship statistics
            rel_stats = session.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                CALL {
                    WITH relationshipType
                    MATCH ()-[r:`${relationshipType}`]->()
                    RETURN count(r) as count
                }
                RETURN relationshipType, count
                ORDER BY relationshipType
            """)
            relationship_counts = {record["relationshipType"]: record["count"] for record in rel_stats}
            
            # Get database size info (if available)
            try:
                size_info = session.run("""
                    CALL dbms.database.size('neo4j') YIELD total, free
                    RETURN total, free
                """).single()
                db_size = {
                    "total": size_info["total"],
                    "free": size_info["free"],
                    "used": size_info["total"] - size_info["free"]
                }
            except:
                db_size = None
            
            return {
                "success": True,
                "message": "Graph statistics retrieved successfully",
                "statistics": {
                    "node_counts": node_counts,
                    "total_nodes": sum(node_counts.values()),
                    "relationship_counts": relationship_counts,
                    "total_relationships": sum(relationship_counts.values()),
                    "database_size": db_size
                }
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to retrieve graph statistics: {str(e)}"
        }

@app.get("/graph/schema")
def get_graph_schema():
    """Get the current schema of the Neo4j database including node labels, relationship types,
    and their properties.
    
    Returns:
        dict: Database schema information
    """
    try:
        with driver.session() as session:
            # Get node labels and their properties
            node_schema = session.run("""
                CALL db.labels() YIELD label
                CALL {
                    WITH label
                    MATCH (n:`${label}`)
                    RETURN DISTINCT keys(n) as properties
                    LIMIT 1
                }
                RETURN label, properties
                ORDER BY label
            """)
            
            # Get relationship types and their properties
            rel_schema = session.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                CALL {
                    WITH relationshipType
                    MATCH ()-[r:`${relationshipType}`]->()
                    RETURN DISTINCT keys(r) as properties
                    LIMIT 1
                }
                RETURN relationshipType, properties
                ORDER BY relationshipType
            """)
            
            # Get indexes
            indexes = session.run("""
                SHOW INDEXES
                YIELD name, labelsOrTypes, properties, type
                RETURN name, labelsOrTypes, properties, type
            """)
            
            return {
                "success": True,
                "message": "Schema information retrieved successfully",
                "schema": {
                    "nodes": {
                        record["label"]: record["properties"]
                        for record in node_schema
                    },
                    "relationships": {
                        record["relationshipType"]: record["properties"]
                        for record in rel_schema
                    },
                    "indexes": [
                        {
                            "name": record["name"],
                            "labels": record["labelsOrTypes"],
                            "properties": record["properties"],
                            "type": record["type"]
                        }
                        for record in indexes
                    ]
                }
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to retrieve schema information: {str(e)}"
        }

@app.get("/graph/search")
def search_entities(
    label: str = None,
    property_name: str = None,
    property_value: str = None,
    limit: int = 10
):
    """Search for entities in the graph database based on label and property values.
    
    Args:
        label (str): Optional node label to filter by
        property_name (str): Name of the property to search
        property_value (str): Value to search for (supports partial matches)
        limit (int): Maximum number of results to return
        
    Returns:
        dict: Matching entities and their properties
    """
    try:
        if not property_name or not property_value:
            return {
                "success": False,
                "message": "property_name and property_value are required parameters"
            }
            
        with driver.session() as session:
            # Build query based on whether label is provided
            if label:
                query = f"""
                MATCH (n:`{label}`)
                WHERE n.`{property_name}` =~ $value
                RETURN n
                LIMIT $limit
                """
            else:
                query = f"""
                MATCH (n)
                WHERE n.`{property_name}` =~ $value
                RETURN n, labels(n) as labels
                LIMIT $limit
                """
            
            # Execute search with case-insensitive regex
            results = session.run(
                query,
                value=f"(?i).*{property_value}.*",
                limit=limit
            )
            
            # Format results
            entities = []
            for record in results:
                node = record["n"]
                entity = {
                    "labels": record["labels"] if "labels" in record else [label],
                    "properties": dict(node.items())
                }
                entities.append(entity)
            
            return {
                "success": True,
                "message": "Search completed successfully",
                "results": {
                    "count": len(entities),
                    "entities": entities
                }
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Search failed: {str(e)}"
        }

