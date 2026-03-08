
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_DIM


class Embedder:
    """Singleton-safe sentence embedding wrapper."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _load(self):
        if self._model is None:
            print(f"[embedder] Loading model: {EMBEDDING_MODEL}")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed one or more texts.

        Returns
        -------
        np.ndarray of shape (N, EMBEDDING_DIM), dtype float32, L2-normalised.
        Normalisation means cosine similarity == dot product – cheaper at
        lookup time and correct for angular distance clustering.
        """
        model = self._load()
        if isinstance(texts, str):
            texts = [texts]

        vecs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   
        )
        return vecs.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string, return 1-D array of shape (DIM,)."""
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM
