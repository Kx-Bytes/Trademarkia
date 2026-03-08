
import threading
import time

import numpy as np
import pytest

from src.cache.semantic_cache import SemanticCache, CacheEntry
from tests.conftest import DIM, K, unit_vec, make_membership




def _store(cache: SemanticCache, seed: int, cluster: int,
           result: str = "result", boundary: bool = False) -> np.ndarray:
    """Store one entry and return its embedding."""
    emb  = unit_vec(seed, DIM)
    mem  = make_membership(cluster, K, boundary=boundary)
    cache.store(emb, f"query-seed-{seed}", result, mem)
    return emb


def _lookup(cache: SemanticCache, emb: np.ndarray, cluster: int,
            boundary: bool = False):
    mem = make_membership(cluster, K, boundary=boundary)
    return cache.lookup(emb, mem)




class TestHitMiss:
    def test_miss_on_empty_cache(self):
        cache = SemanticCache(threshold=0.90)
        result = _lookup(cache, unit_vec(0), cluster=0)
        assert result is None

    def test_exact_match_is_hit(self):
        cache = SemanticCache(threshold=0.90)
        emb   = _store(cache, seed=0, cluster=0, result="space news")
        hit   = _lookup(cache, emb, cluster=0)
        assert hit is not None
        entry, sim = hit
        assert sim == pytest.approx(1.0, abs=1e-5)
        assert entry.result == "space news"

    def test_dissimilar_vector_is_miss(self):
        """Two orthogonal unit vectors → cosine similarity 0 → miss."""
        cache = SemanticCache(threshold=0.90)
        _store(cache, seed=0, cluster=0)
        v0  = unit_vec(0, DIM)
        orth = np.zeros(DIM, dtype=np.float32)
        orth[1] = 1.0
        orth -= np.dot(orth, v0) * v0          
        orth /= np.linalg.norm(orth)
        result = _lookup(cache, orth, cluster=0)
        assert result is None

    def test_below_threshold_is_miss(self):
        """Store at threshold=0.99; lookup with a vector that is similar
        but not identical → should miss under the high threshold."""
        cache = SemanticCache(threshold=0.99)
        emb   = _store(cache, seed=5, cluster=1)
        noise = np.random.default_rng(99).standard_normal(DIM).astype(np.float32) * 0.3
        perturbed = emb + noise
        perturbed /= np.linalg.norm(perturbed)
        result = _lookup(cache, perturbed, cluster=1)
        assert result is None

    def test_threshold_zero_always_hits(self):
        """threshold=0.0 → any stored entry matches anything."""
        cache = SemanticCache(threshold=0.0)
        _store(cache, seed=0, cluster=0)
        result = _lookup(cache, unit_vec(99), cluster=0)
        assert result is not None

    def test_threshold_one_self_match_only(self):
        """
        Near-1.0 threshold: only the same vector matches.
        We use 0.9999 not 1.0 because np.dot(v_f32, v_f32.copy()) can return
        ~0.99999994 due to float32 rounding, falling just below a strict 1.0
        threshold and producing a false miss.
        """
        cache = SemanticCache(threshold=0.9999)
        emb   = _store(cache, seed=0, cluster=0)
        # Self-match -> hit
        assert _lookup(cache, emb.copy(), cluster=0) is not None
        # Clearly different vector -> miss
        assert _lookup(cache, unit_vec(99), cluster=0) is None

    def test_hit_returns_best_similarity(self):
        """When two stored entries both pass threshold, the closer one wins."""
        cache = SemanticCache(threshold=0.50)
        emb_a = unit_vec(0, DIM)
        emb_b = unit_vec(1, DIM)
        cache.store(emb_a, "q-a", "result-a", make_membership(0, K))
        cache.store(emb_b, "q-b", "result-b", make_membership(0, K))

        hit = _lookup(cache, emb_a, cluster=0)
        assert hit is not None
        entry, sim = hit
        assert entry.result == "result-a"
        assert sim == pytest.approx(1.0, abs=1e-5)



class TestBuckets:
    def test_entry_stored_in_dominant_bucket(self):
        cache = SemanticCache(threshold=0.90)
        _store(cache, seed=0, cluster=2)
        assert len(cache._buckets[2]) == 1
        assert all(len(cache._buckets[c]) == 0 for c in [0, 1, 3])

    def test_boundary_entry_in_two_buckets(self):
        """A boundary entry (2nd membership > 0.15) appears in 2 buckets."""
        cache = SemanticCache(threshold=0.90)
        emb   = unit_vec(0, DIM)
        mem   = make_membership(0, K, boundary=True)   
        cache.store(emb, "q", "r", mem)

        second = int(np.argsort(mem)[::-1][1])
        assert len(cache._buckets[0]) == 1
        assert len(cache._buckets[second]) == 1
        assert cache._total_entries() == 1

    def test_boundary_entry_findable_from_second_cluster(self):
        """Lookup routed via second cluster should still find the entry."""
        cache = SemanticCache(threshold=0.90)
        emb   = unit_vec(0, DIM)
        mem   = make_membership(0, K, boundary=True)
        cache.store(emb, "q-boundary", "r-boundary", mem)

        second = int(np.argsort(mem)[::-1][1])
        query_mem = make_membership(second, K, boundary=False)
        hit = cache.lookup(emb, query_mem)
        assert hit is not None
        assert hit[0].original_query == "q-boundary"

    def test_non_boundary_entry_not_in_second_bucket(self):
        """Non-boundary entry (2nd < 0.15) must NOT appear in a second bucket."""
        cache = SemanticCache(threshold=0.90)
        mem   = make_membership(1, K, boundary=False)  
        cache.store(unit_vec(0), "q", "r", mem)
        second = int(np.argsort(mem)[::-1][1])
        assert len(cache._buckets[second]) == 0



class TestLRUEviction:
    def test_eviction_triggered_at_max_size(self):
        max_size = 5
        cache = SemanticCache(threshold=0.90, max_size=max_size)
        for i in range(max_size + 1):
            _store(cache, seed=i * 10, cluster=i % K)
        assert cache._total_entries() <= max_size

    def test_lru_oldest_is_evicted(self):
        """
        Insert entries with a small sleep between them.
        The first-inserted entry should be evicted when capacity is exceeded.
        """
        cache = SemanticCache(threshold=0.90, max_size=3)

        emb_old = _store(cache, seed=0, cluster=0, result="old")
        time.sleep(0.01)
        _store(cache, seed=10, cluster=1, result="middle")
        time.sleep(0.01)
        _store(cache, seed=20, cluster=2, result="newer")
        time.sleep(0.01)
        _store(cache, seed=30, cluster=3, result="newest")

        hit = _lookup(cache, emb_old, cluster=0)
        assert hit is None

    def test_accessed_entry_survives_eviction(self):
        """Accessing an old entry refreshes its LRU timestamp."""
        cache = SemanticCache(threshold=0.90, max_size=3)

        emb_first = _store(cache, seed=0, cluster=0, result="first")
        time.sleep(0.01)
        _store(cache, seed=10, cluster=1)
        time.sleep(0.01)
        _lookup(cache, emb_first, cluster=0)
        time.sleep(0.01)
        _store(cache, seed=20, cluster=2)
        time.sleep(0.01)
        _store(cache, seed=30, cluster=3)

        assert _lookup(cache, emb_first, cluster=0) is not None

    def test_boundary_entry_evicted_from_both_buckets(self):
        """Evicting a boundary entry must remove it from both its buckets."""
        cache = SemanticCache(threshold=0.90, max_size=2)

        emb_b = unit_vec(0, DIM)
        mem_b = make_membership(0, K, boundary=True)
        second = int(np.argsort(mem_b)[::-1][1])   

        time.sleep(0.01)
        cache.store(emb_b, "boundary-q", "r", mem_b)  


        other_clusters = [c for c in range(K) if c not in (0, second)]
        time.sleep(0.01)
        _store(cache, seed=10, cluster=other_clusters[0])
        time.sleep(0.01)
        _store(cache, seed=20, cluster=other_clusters[1])   

        assert len(cache._buckets[0]) == 0
        assert len(cache._buckets[second]) == 0



class TestFlushAndStats:
    def test_flush_empties_cache(self):
        cache = SemanticCache(threshold=0.90)
        _store(cache, 0, 0)
        _store(cache, 10, 1)
        cache.flush()
        assert cache._total_entries() == 0
        assert all(len(v) == 0 for v in cache._buckets.values())

    def test_flush_resets_counters(self):
        cache = SemanticCache(threshold=0.90)
        emb = _store(cache, 0, 0)
        _lookup(cache, emb, 0)   
        _lookup(cache, unit_vec(99), 0)  
        cache.flush()
        s = cache.stats
        assert s["hit_count"]  == 0
        assert s["miss_count"] == 0
        assert s["hit_rate"]   == 0.0

    def test_stats_zero_queries(self):
        cache = SemanticCache(threshold=0.90)
        s = cache.stats
        assert s["hit_rate"] == 0.0   

    def test_stats_arithmetic(self):
        cache = SemanticCache(threshold=0.90)
        emb = _store(cache, 0, 0)
        _lookup(cache, emb, 0)                # hit
        _lookup(cache, emb, 0)                # hit
        _lookup(cache, unit_vec(50), 0)       # miss
        s = cache.stats
        assert s["hit_count"]  == 2
        assert s["miss_count"] == 1
        assert s["hit_rate"]   == pytest.approx(2/3, abs=0.001)

    def test_total_entries_counts_unique(self):
        """A boundary entry stored in 2 buckets still counts as 1 unique entry."""
        cache = SemanticCache(threshold=0.90)
        emb  = unit_vec(0, DIM)
        mem  = make_membership(0, K, boundary=True)
        cache.store(emb, "q", "r", mem)
        assert cache.stats["total_entries"] == 1

    def test_multiple_entries_counted(self):
        cache = SemanticCache(threshold=0.90)
        for i in range(7):
            _store(cache, seed=i * 7, cluster=i % K)
        assert cache.stats["total_entries"] == 7




class TestThreadSafety:
    def test_concurrent_stores_no_race(self):
        """200 concurrent stores must not corrupt internal state."""
        cache  = SemanticCache(threshold=0.90, max_size=500)
        errors = []

        def worker(seed):
            try:
                _store(cache, seed=seed, cluster=seed % K)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in threads: {errors}"
        assert cache._total_entries() <= 500

    def test_concurrent_lookups_no_race(self):
        """Concurrent reads against a populated cache must not raise."""
        cache = SemanticCache(threshold=0.90)
        emb   = _store(cache, seed=0, cluster=0)
        errors = []

        def reader(_):
            try:
                _lookup(cache, emb, cluster=0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
