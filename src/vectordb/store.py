

from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, CHROMA_COLLECTION


class VectorStore:
    """Persistent ChromaDB collection with convenience helpers."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},  # cosine distance index
        )

    # ── Write ────────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Batch-upsert documents.  Safe to call repeatedly (idempotent)."""
        self._col.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    # ── Read ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Return the n_results nearest neighbours.

        Parameters
        ----------
        where : optional ChromaDB metadata filter, e.g.
                {"dominant_cluster": {"$eq": 3}}
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self._col.query(**kwargs)

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        return self._col.get(ids=ids, include=["documents", "metadatas"])

    def count(self) -> int:
        return self._col.count()

    def is_populated(self) -> bool:
        return self.count() > 0
