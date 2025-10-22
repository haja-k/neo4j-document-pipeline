import yaml
from neo4j import GraphDatabase

with open("config/neo4jConfig.yaml", "r") as f:
    config = yaml.safe_load(f)

NEO4J_URI = config["uri"]
NEO4J_USER = config["user"]
NEO4J_PASSWORD = config["password"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
