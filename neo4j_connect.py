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
MAX_CONNECTION_POOL_SIZE = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "100"))
CONNECTION_ACQUISITION_TIMEOUT = int(os.getenv("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", "30"))

# Configure driver with larger connection pool and timeouts
driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
    connection_acquisition_timeout=CONNECTION_ACQUISITION_TIMEOUT,
    max_connection_lifetime=3600,  # 1 hour
    connection_timeout=30,  # 30 seconds to establish connection
    keep_alive=True,
    encrypted=False  # Set to True if using SSL
)

def get_driver():
    """Get the Neo4j driver instance"""
    return driver

def close_driver():
    """Close the Neo4j driver (call on app shutdown)"""
    driver.close()

