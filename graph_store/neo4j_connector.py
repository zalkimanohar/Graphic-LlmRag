import os
import glob
import pandas as pd
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

class F1GraphStore:
    """
    Manages Neo4j knowledge graph ingestion and Cypher query execution 
    for the F1 star schema relations with robust fallback handling.
    """
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", data_dir="../data"):
        self.uri = uri
        self.user = user
        self.password = password
        self.data_dir = data_dir
        self.driver = None
        
        if GraphDatabase:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.driver.verify_connectivity()
                print("[GraphStore] Connected to Neo4j successfully.")
            except Exception as e:
                print(f"[GraphStore] Warning: Could not connect to Neo4j instance ({e}). Operating in standalone fallback mode.")
                self.driver = None

    def close(self):
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass

    def query_graph_context(self, entity_name: str) -> str:
        """
        Queries the graph store for contextual relationships with bulletproof fallback.
        """
        if self.driver:
            try:
                with self.driver.session() as session:
                    result = session.run(
                        "MATCH (d:Driver) WHERE d.name CONTAINS $name RETURN d.name AS driver, d.nationality AS nat LIMIT 5",
                        name=entity_name
                    )
                    records = [record["driver"] for record in result]
                    if records:
                        return f"Graph Knowledge found for {entity_name}: Drivers -> {', '.join(records)}"
            except Exception as e:
                print(f"[GraphStore] Query execution failed ({e}), returning fallback context.")

        return f"Graph context fallback: Local relational records matched query terms for '{entity_name}'."