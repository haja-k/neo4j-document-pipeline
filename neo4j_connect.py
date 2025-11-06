import yaml
import os
from neo4j import GraphDatabase

with open("config/neo4jConfig.yaml", "r") as f:
    config = yaml.safe_load(f)

# Allow environment variable override for Docker
NEO4J_URI = os.getenv("NEO4J_URI", config["uri"])
NEO4J_USER = os.getenv("NEO4J_USER", config["user"])
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", config["password"])

# Connection pooling configuration for high concurrency
MAX_CONNECTION_POOL_SIZE = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50"))
CONNECTION_ACQUISITION_TIMEOUT = int(os.getenv("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", "30"))

driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
    connection_acquisition_timeout=CONNECTION_ACQUISITION_TIMEOUT,
    max_connection_lifetime=3600,  # 1 hour
    keep_alive=True
)

