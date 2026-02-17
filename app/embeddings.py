from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingManager:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded.")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
