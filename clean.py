from neo4j_connect import driver
from neo4j.exceptions import CypherSyntaxError

def _drop_all_constraints_and_indexes(session):
    cons = session.run("SHOW CONSTRAINTS YIELD name RETURN name").value()
    for name in cons:
        session.run(f"DROP CONSTRAINT {name} IF EXISTS")
    print(f"🧽 Dropped {len(cons)} constraints")

    idxs = session.run("SHOW INDEXES YIELD name RETURN name").value()
    for name in idxs:
        session.run(f"DROP INDEX {name} IF EXISTS")
    print(f"🧽 Dropped {len(idxs)} indexes")

def _delete_all_nodes_subquery_in_tx(session, batch_size: int):
    # Neo4j 5+ syntax; some deployments require a trailing RETURN
    cypher = f"""
    CALL {{
      WITH 1 AS _
      MATCH (n)
      WITH n LIMIT {batch_size}
      DETACH DELETE n
    }} IN TRANSACTIONS OF {batch_size} ROWS
    RETURN 0
    """
    session.run(cypher)
    print("🧨 Deleted all nodes/relationships via subquery-in-transactions")

def _delete_all_nodes_apoc(session, batch_size: int):
    session.run("""
    CALL apoc.periodic.iterate(
      'MATCH (n) RETURN n',
      'DETACH DELETE n',
      {batchSize: $bs, parallel: true}
    )
    """, {"bs": batch_size})
    print("🧨 Deleted all nodes/relationships via APOC periodic.iterate")

def _delete_all_nodes_loop(session, batch_size: int):
    total = 0
    while True:
        deleted = session.run(f"""
        MATCH (n)
        WITH n LIMIT {batch_size}
        DETACH DELETE n
        RETURN count(*) AS c
        """).single()["c"]
        total += deleted
        if deleted == 0:
            break
    print(f"🧨 Deleted all nodes/relationships via loop (total {total})")

def nuke_knowledge_base(batch_size: int = 100_000):
    with driver.session() as s:
        _drop_all_constraints_and_indexes(s)

        # Try Neo4j 5 syntax first
        try:
            _delete_all_nodes_subquery_in_tx(s, batch_size)
            return
        except CypherSyntaxError:
            pass  # fall through

        # Try APOC
        try:
            _delete_all_nodes_apoc(s, batch_size)
            return
        except Exception:
            pass  # fall through

        # Final fallback: simple loop
        _delete_all_nodes_loop(s, batch_size)

if __name__ == "__main__":
    nuke_knowledge_base()
    print("✅ Knowledge base fully cleared.")
