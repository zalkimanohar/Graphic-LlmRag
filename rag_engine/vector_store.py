import os
try:
    import chromadb
except ImportError:
    chromadb = None

class F1VectorStore:
    """
    Local persistent vector database using ChromaDB for F1 knowledge indexing.
    """
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            # Robustly resolve path relative to this file's directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            persist_directory = os.path.join(base_dir, "chroma_db")
            
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        
        if chromadb:
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                self.collection = self.client.get_or_create_collection(name="f1_knowledge_base")
                print(f"[VectorStore] Initialized ChromaDB at {self.persist_directory}")
            except Exception as e:
                print(f"[VectorStore] Warning: Could not initialize ChromaDB ({e}).")

    def add_documents(self, documents: list, metadatas: list, ids: list):
        if self.collection:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"[VectorStore] Indexed {len(documents)} documents successfully.")
            except Exception as e:
                print(f"[VectorStore] Error adding documents: {e}")
        else:
            print("[VectorStore] ChromaDB collection inactive. Skipping vector insertion.")

    def query(self, query_texts: list, n_results: int = 3) -> dict:
        if self.collection:
            try:
                return self.collection.query(query_texts=query_texts, n_results=n_results)
            except Exception as e:
                print(f"[VectorStore] Query execution error: {e}")
        return {"documents": [["Fallback F1 context document record."]], "metadatas": [[{}]]}