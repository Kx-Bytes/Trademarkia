"""
tests/test_api.py

Integration tests for the FastAPI endpoints (POST /query, GET /cache/stats,
DELETE /cache, GET /health).

All heavy dependencies (Embedder, FuzzyCMeans, VectorStore) are replaced with
lightweight fakes via the `api_client` fixture in conftest.py.
No prepare_data.py run is required.

Covers
──────
POST /query
  - First call is a cache miss (cache_hit=False)
  - Second identical call is a hit (cache_hit=True)
  - Hit returns matched_query and similarity_score
  - Miss has null matched_query and null similarity_score
  - Response includes dominant_cluster integer
  - Response includes non-empty result string
  - Empty query string returns 422
  - Missing query field returns 422
  - VectorStore query is called on a miss, not called on a hit

GET /cache/stats
  - Returns all required fields
  - hit_rate is 0 before any queries
  - Counters update correctly after hits and misses

DELETE /cache
  - Returns 200 with entries_cleared
  - Cache is actually empty after flush
  - Stats reset to zero after flush

GET /health
  - Returns 200
  - Contains vector_db_docs and cache_entries fields
"""

import pytest
from fastapi.testclient import TestClient

# The api_client fixture is in conftest.py and injected automatically


# ──────────────────────────────────────────────────────────────────────────────
# POST /query
# ──────────────────────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_first_query_is_cache_miss(self, api_client):
        client, cache = api_client
        resp = client.post("/query", json={"query": "Tell me about space exploration"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_hit"] is False

    def test_second_identical_query_is_hit(self, api_client):
        client, cache = api_client
        q = {"query": "Tell me about space exploration"}
        client.post("/query", json=q)       # miss – populates cache
        resp = client.post("/query", json=q)  # hit
        assert resp.status_code == 200
        assert resp.json()["cache_hit"] is True

    def test_hit_has_matched_query_and_similarity(self, api_client):
        client, _ = api_client
        q = {"query": "Tell me about NASA"}
        client.post("/query", json=q)
        resp = client.post("/query", json=q)
        data = resp.json()
        assert data["matched_query"] is not None
        assert data["similarity_score"] is not None
        assert 0.0 <= data["similarity_score"] <= 1.0

    def test_miss_has_null_matched_query(self, api_client):
        client, _ = api_client
        resp = client.post("/query", json={"query": "Unique query one"})
        data = resp.json()
        assert data["matched_query"] is None
        assert data["similarity_score"] is None

    def test_response_has_dominant_cluster(self, api_client):
        client, _ = api_client
        resp = client.post("/query", json={"query": "any query"})
        data = resp.json()
        assert "dominant_cluster" in data
        assert isinstance(data["dominant_cluster"], int)

    def test_response_has_nonempty_result(self, api_client):
        client, _ = api_client
        resp = client.post("/query", json={"query": "space shuttle"})
        data = resp.json()
        assert "result" in data
        assert len(data["result"]) > 0

    def test_empty_query_returns_422(self, api_client):
        client, _ = api_client
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_field_returns_422(self, api_client):
        client, _ = api_client
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_vectorstore_called_on_miss(self, api_client, mock_app_state):
        client, _ = api_client
        mock_app_state["store"].query.reset_mock()
        client.post("/query", json={"query": "fresh unique query abc123"})
        # On a miss the store.query must be called
        mock_app_state["store"].query.assert_called()

    def test_vectorstore_not_called_on_hit(self, api_client, mock_app_state):
        client, _ = api_client
        q = {"query": "cached query xyz"}
        client.post("/query", json=q)          # miss
        mock_app_state["store"].query.reset_mock()
        client.post("/query", json=q)          # hit – should NOT call store
        mock_app_state["store"].query.assert_not_called()

    def test_response_query_field_matches_input(self, api_client):
        client, _ = api_client
        my_query = "What happened to the Atlantis shuttle?"
        resp = client.post("/query", json={"query": my_query})
        assert resp.json()["query"] == my_query


# ──────────────────────────────────────────────────────────────────────────────
# GET /cache/stats
# ──────────────────────────────────────────────────────────────────────────────

class TestCacheStats:
    def test_stats_returns_200(self, api_client):
        client, _ = api_client
        assert client.get("/cache/stats").status_code == 200

    def test_stats_has_required_fields(self, api_client):
        client, _ = api_client
        data = client.get("/cache/stats").json()
        for field in ("total_entries", "hit_count", "miss_count", "hit_rate"):
            assert field in data, f"Missing field: {field}"

    def test_initial_hit_rate_zero(self, api_client):
        client, _ = api_client
        assert client.get("/cache/stats").json()["hit_rate"] == 0.0

    def test_miss_count_increments(self, api_client):
        client, _ = api_client
        client.post("/query", json={"query": "miss 1"})
        client.post("/query", json={"query": "miss 2"})
        stats = client.get("/cache/stats").json()
        assert stats["miss_count"] >= 2

    def test_hit_count_increments(self, api_client):
        client, _ = api_client
        q = {"query": "repeated question"}
        client.post("/query", json=q)    # miss
        client.post("/query", json=q)    # hit
        client.post("/query", json=q)    # hit
        stats = client.get("/cache/stats").json()
        assert stats["hit_count"] >= 2

    def test_hit_rate_correct_after_queries(self, api_client):
        client, _ = api_client
        q = {"query": "consistent question"}
        client.post("/query", json=q)    # miss
        client.post("/query", json=q)    # hit
        stats = client.get("/cache/stats").json()
        # 1 hit, 1 miss → hit_rate = 0.5
        assert stats["hit_rate"] == pytest.approx(0.5, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /cache
# ──────────────────────────────────────────────────────────────────────────────

class TestCacheFlush:
    def test_flush_returns_200(self, api_client):
        client, _ = api_client
        assert client.delete("/cache").status_code == 200

    def test_flush_reports_entries_cleared(self, api_client):
        client, _ = api_client
        client.post("/query", json={"query": "entry to clear"})
        resp = client.delete("/cache")
        data = resp.json()
        assert "entries_cleared" in data
        assert data["entries_cleared"] >= 1

    def test_flush_empties_cache(self, api_client):
        client, _ = api_client
        client.post("/query", json={"query": "something"})
        client.delete("/cache")
        stats = client.get("/cache/stats").json()
        assert stats["total_entries"] == 0

    def test_flush_resets_stats(self, api_client):
        client, _ = api_client
        q = {"query": "reset me"}
        client.post("/query", json=q)
        client.post("/query", json=q)
        client.delete("/cache")
        stats = client.get("/cache/stats").json()
        assert stats["hit_count"]  == 0
        assert stats["miss_count"] == 0
        assert stats["hit_rate"]   == 0.0

    def test_post_flush_query_is_miss(self, api_client):
        """After a flush, a previously-cached query should miss again."""
        client, _ = api_client
        q = {"query": "flush then re-query"}
        client.post("/query", json=q)    # populates cache
        client.delete("/cache")
        resp = client.post("/query", json=q)   # should miss again
        assert resp.json()["cache_hit"] is False


# ──────────────────────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, api_client):
        client, _ = api_client
        assert client.get("/health").status_code == 200

    def test_health_has_vector_db_docs(self, api_client):
        client, _ = api_client
        data = client.get("/health").json()
        assert "vector_db_docs" in data
        assert isinstance(data["vector_db_docs"], int)

    def test_health_has_cache_entries(self, api_client):
        client, _ = api_client
        data = client.get("/health").json()
        assert "cache_entries" in data

    def test_health_has_status_ok(self, api_client):
        client, _ = api_client
        assert client.get("/health").json()["status"] == "ok"
