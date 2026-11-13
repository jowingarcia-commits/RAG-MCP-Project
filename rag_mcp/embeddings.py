"""Text to vectors with sentence-transformers. The model downloads on first use."""
from functools import lru_cache

import numpy as np

from config.settings import EMBEDDING_MODEL as MODEL_NAME
DIM = 384  # output size of all-MiniLM-L6-v2; change it if you change the model


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """Unit-length float32 vectors, so inner product equals cosine similarity."""
    vecs = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.ascontiguousarray(vecs, dtype="float32")


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    text = " ".join(text.split())
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]
