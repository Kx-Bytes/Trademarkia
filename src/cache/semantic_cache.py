
from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import CACHE_SIMILARITY_THRESHOLD, CACHE_MAX_SIZE


@dataclass
class CacheEntry:
    query_embedding:  np.ndarray          
    original_query:   str
    result:           str
    membership:       np.ndarray         
    dominant_cluster: int
    access_time:      float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update access time (LRU)."""
        self.access_time = time.time()


class SemanticCache:
    """
    Thread-safe semantic cache with cluster-bucketed lookup and LRU eviction.

    Parameters
    ----------
    threshold : float
        Cosine similarity threshold for a hit.  See module docstring.
    max_size  : int
        Maximum total entries before LRU eviction.
    """

    def __init__(
        self,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        max_size:  int   = CACHE_MAX_SIZE,
    ):
        self.threshold = threshold
        self.max_size  = max_size

        self._buckets: Dict[int, List[CacheEntry]] = defaultdict(list)

        self._hit_count:  int = 0
        self._miss_count: int = 0

        self._lock = threading.RLock()


    def lookup(
        self,
        query_embedding: np.ndarray,
        membership:      np.ndarray,
    ) -> Optional[Tuple[CacheEntry, float]]:
        """
        Search for a cached entry similar to query_embedding.

        Parameters
        ----------
        query_embedding : (D,) unit-norm vector
        membership      : (k,) soft cluster membership for the query

        Returns
        -------
        (entry, similarity) if hit, else None.
        Updates hit/miss counter.
        """
        with self._lock:
            result = self._search(query_embedding, membership)
            if result is not None:
                entry, sim = result
                entry.touch()
                self._hit_count += 1
                return entry, sim
            else:
                self._miss_count += 1
                return None

    def store(
        self,
        query_embedding:  np.ndarray,
        original_query:   str,
        result:           str,
        membership:       np.ndarray,
    ) -> None:
        """
        Add a new entry to the cache.

        The entry is placed in the bucket of its dominant cluster.
        Boundary entries (second membership > 0.15) are also placed in
        the second-best bucket so they are found by either route.
        """
        with self._lock:
            dominant = int(np.argmax(membership))
            entry = CacheEntry(
                query_embedding  = query_embedding.copy(),
                original_query   = original_query,
                result           = result,
                membership       = membership.copy(),
                dominant_cluster = dominant,
            )
            self._buckets[dominant].append(entry)

            sorted_idx = np.argsort(membership)[::-1]
            if len(sorted_idx) > 1 and membership[sorted_idx[1]] > 0.15:
                second = int(sorted_idx[1])
                self._buckets[second].append(entry)

            if self._total_entries() > self.max_size:
                self._evict_lru()

    def flush(self) -> None:
        """Clear all entries and reset stats."""
        with self._lock:
            self._buckets.clear()
            self._hit_count  = 0
            self._miss_count = 0

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total   = self._total_entries()
            hits    = self._hit_count
            misses  = self._miss_count
            total_q = hits + misses
            return {
                "total_entries": total,
                "hit_count":     hits,
                "miss_count":    misses,
                "hit_rate":      round(hits / total_q, 4) if total_q else 0.0,
            }


    def _search(
        self,
        q_emb:      np.ndarray,
        membership: np.ndarray,
    ) -> Optional[Tuple[CacheEntry, float]]:
        """
        Scan the 1–2 relevant buckets.  Return best hit above threshold.
        Called under self._lock.
        """
        sorted_clusters = np.argsort(membership)[::-1]
        clusters_to_search = [int(sorted_clusters[0])]
        if (
            len(sorted_clusters) > 1
            and membership[sorted_clusters[1]] > 0.15
        ):
            clusters_to_search.append(int(sorted_clusters[1]))

        best_entry:  Optional[CacheEntry] = None
        best_sim:    float                = -1.0

        for c in clusters_to_search:
            for entry in self._buckets.get(c, []):
                sim = float(np.dot(q_emb, entry.query_embedding))
                if sim >= self.threshold and sim > best_sim:
                    best_sim   = sim
                    best_entry = entry

        return (best_entry, best_sim) if best_entry is not None else None

    def _total_entries(self) -> int:
        """Count unique entries across all buckets (de-dup by identity)."""
        seen = set()
        count = 0
        for entries in self._buckets.values():
            for e in entries:
                if id(e) not in seen:
                    seen.add(id(e))
                    count += 1
        return count

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry from its bucket(s)."""
        oldest:       Optional[CacheEntry] = None
        oldest_time:  float                = float("inf")

        for entries in self._buckets.values():
            for e in entries:
                if e.access_time < oldest_time:
                    oldest_time = e.access_time
                    oldest      = e

        if oldest is None:
            return

        for bucket in self._buckets.values():
            to_remove = [e for e in bucket if e is oldest]
            for e in to_remove:
                bucket.remove(e)
