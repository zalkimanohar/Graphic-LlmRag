try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class F1Embedder:
    """
    Local embedding generator using lightweight, high-performance open-source models.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer:
            try:
                print(f"[Embedder] Loading local embedding model: {model_name}...")
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"[Embedder] Warning: Could not load sentence-transformers ({e}).")

    def encode(self, texts: list) -> list:
        if self.model:
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        # Fallback dummy embedding vector if model unavailable
        return [[0.1] * 384 for _ in texts]