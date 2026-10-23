import os
try:
    import chromadb
except ImportError:
    chromadb = None

class F1VectorStore:
    """
    Local persistent vector database using ChromaDB for F1 knowledge indexing.
    """
    def __init__(self, persist_directory: str = "rag_engine/chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        
        if chromadb:
            try:
                os.makedirs(persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_directory)
                self.collection = self.client.get_or_create_collection(name="f1_knowledge_base")
                print(f"[VectorStore] Initialized ChromaDB at {persist_directory}")
            except Exception as e:
                print(f"[VectorStore] Warning: Could not initialize ChromaDB ({e}).")

    def add_documents(self, documents: list, metadatas: list, ids: list):
        if self.collection:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[VectorStore] Indexed {len(documents)} documents successfully.")
        else:
            print("[VectorStore] ChromaDB collection inactive. Skipping vector insertion.")

    def query(self, query_texts: list, n_results: int = 3) -> dict:
        if self.collection:
            return self.collection.query(query_texts=query_texts, n_results=n_results)
        return {"documents": [["Fallback F1 context document record."]], "metadatas": [[{}]]}