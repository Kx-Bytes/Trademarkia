"""
tests/conftest.py – Shared pytest fixtures.

All heavy objects (Embedder, FuzzyCMeans, VectorStore) are replaced with
lightweight fakes so the test suite runs in seconds without requiring
`prepare_data.py` to have been executed first.
"""

import hashlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make sure the repo root is on sys.path regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Dimensionality constants (kept tiny for speed)
# ──────────────────────────────────────────────────────────────────────────────
DIM = 8    # embedding dimension used in all fake vectors
K   = 4    # number of fuzzy clusters used in all fake models


def unit_vec(seed: int = 0, dim: int = DIM) -> np.ndarray:
    """Return a deterministic unit-norm vector."""
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_membership(dominant: int, k: int = K, boundary: bool = False) -> np.ndarray:
    """
    Create a soft membership vector.

    dominant  : cluster that receives the most weight
    boundary  : if True, second cluster gets > 0.15 (triggers dual-bucket indexing)
    """
    u = np.zeros(k, dtype=np.float32)
    if boundary:
        u[dominant]               = 0.60
        u[(dominant + 1) % k]     = 0.25
        u[(dominant + 2) % k]     = 0.10
        u[(dominant + 3) % k]     = 0.05
    else:
        u[dominant]               = 0.85
        remaining = np.ones(k - 1, dtype=np.float32) * (0.15 / (k - 1))
        others = [i for i in range(k) if i != dominant]
        for idx, c in enumerate(others):
            u[c] = remaining[idx]
    # Normalise to sum-to-1 (floating-point safety)
    u /= u.sum()
    return u


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI test client fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_app_state():
    """
    Returns a dict of patched app-state objects for use with the FastAPI app.
    """
    from src.cache.semantic_cache import SemanticCache
    from src.clustering.fuzzy_cluster import FuzzyCMeans

    # Fake embedder: deterministic but UNIQUE per query text.
    # Same text -> same vector (cache-hit tests work correctly).
    # Different texts -> different vectors (miss-count tests work correctly).
    # Bug fix: previously always returned unit_vec(seed=0), so two different
    # queries would collide in the cache and the second appeared as a hit.
    def _embed(text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)
        return unit_vec(seed=seed, dim=DIM)

    fake_embedder = MagicMock()
    fake_embedder.embed_one.side_effect = _embed

    # Fake FCM: always returns make_membership(0)
    fake_fcm = MagicMock(spec=FuzzyCMeans)
    fake_fcm.predict_memberships.side_effect = (
        lambda X: np.stack([make_membership(0)] * len(X))
    )

    # Fake VectorStore: returns canned search results
    fake_store = MagicMock()
    fake_store.count.return_value = 100
    fake_store.is_populated.return_value = True
    fake_store.query.return_value = {
        "documents": [["Doc A about space.", "Doc B about rockets."]],
        "distances": [[0.10, 0.20]],
        "metadatas": [[{"dominant_cluster": 0}, {"dominant_cluster": 0}]],
    }

    # Real SemanticCache (we want to test its real behaviour)
    real_cache = SemanticCache(threshold=0.90, max_size=50)

    return {
        "embedder": fake_embedder,
        "fcm":      fake_fcm,
        "store":    fake_store,
        "cache":    real_cache,
    }


@pytest.fixture()
def api_client(mock_app_state):
    """
    TestClient with all heavy dependencies injected via app.state.
    Lifespan is patched out so no model download or disk check happens.
    """
    from contextlib import asynccontextmanager
    from fastapi.testclient import TestClient
    from src.api.main import app

    # Replace the real lifespan (which downloads models) with a no-op
    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan

    # Inject fakes directly onto app.state
    app.state.embedder = mock_app_state["embedder"]
    app.state.fcm      = mock_app_state["fcm"]
    app.state.store    = mock_app_state["store"]
    app.state.cache    = mock_app_state["cache"]

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client, mock_app_state["cache"]

    # Restore original lifespan so other test runs are unaffected
    app.router.lifespan_context = original_lifespan
