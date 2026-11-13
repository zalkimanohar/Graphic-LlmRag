from .vector_store import F1VectorStore
from graph_store.neo4j_connector import F1GraphStore

class F1HybridRetriever:
    """
    Hybrid retriever combining vector semantic search (ChromaDB) 
    and structured entity lookups (Neo4j Graph Store).
    """
    def __init__(self, vector_store: F1VectorStore, graph_store: F1GraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store

    def retrieve_context(self, query: str) -> str:
        # 1. Vector DB Semantic Search
        vector_results = self.vector_store.query(query_texts=[query], n_results=2)
        docs = vector_results.get("documents", [[]])[0]
        vector_context = "\n".join([f"- {doc}" for doc in docs]) if docs else "No specific vector context found."

        # 2. Graph Store Entity Search
        graph_context = self.graph_store.query_graph_context(query)

        return f"""=== VECTOR KNOWLEDGE BASE ===
{vector_context}

=== GRAPH KNOWLEDGE STORE ===
- {graph_context}
"""