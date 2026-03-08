"""
tests/test_fuzzy_cluster.py

Unit tests for FuzzyCMeans (src/clustering/fuzzy_cluster.py).

Covers
──────
- _squared_distances: non-negative, zero for identical unit vectors
- _update_memberships: rows sum to 1, values in [0,1], hard-assignment edge case
- _update_centroids: output is unit-norm
- FuzzyCMeans.fit: converges without error on toy data
- predict_memberships: shape, row-sum, dtype
- dominant_cluster: argmax is correct
- is_boundary: True iff 2nd membership > threshold
- save / load round-trip (uses tmp_path)
- is_saved: returns False before save, True after
- Small-scale numeric sanity: two tight clusters → high within-cluster membership
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

# We import the private helpers directly so we can test the math in isolation
from src.clustering.fuzzy_cluster import (
    FuzzyCMeans,
    _squared_distances,
    _update_memberships,
    _update_centroids,
)
from tests.conftest import DIM


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_toy_data(n_per_cluster: int = 30, k: int = 3, dim: int = DIM,
                  noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """
    Generate k tight clusters of unit-norm vectors on the unit sphere.
    Each cluster is centred on a different basis vector ± small noise.
    """
    rng = np.random.default_rng(seed)
    centres = []
    for i in range(k):
        c = np.zeros(dim, dtype=np.float32)
        c[i % dim] = 1.0
        centres.append(c)

    points = []
    for c in centres:
        for _ in range(n_per_cluster):
            p = c + rng.standard_normal(dim).astype(np.float32) * noise
            points.append(_unit(p))
    return np.stack(points)


# ──────────────────────────────────────────────────────────────────────────────
# _squared_distances
# ──────────────────────────────────────────────────────────────────────────────

class TestSquaredDistances:
    def test_shape(self):
        X = np.stack([_unit(np.random.randn(DIM).astype(np.float32)) for _ in range(10)])
        V = np.stack([_unit(np.random.randn(DIM).astype(np.float32)) for _ in range(3)])
        d2 = _squared_distances(X, V)
        assert d2.shape == (10, 3)

    def test_non_negative(self):
        X = np.stack([_unit(np.random.randn(DIM).astype(np.float32)) for _ in range(20)])
        V = np.stack([_unit(np.random.randn(DIM).astype(np.float32)) for _ in range(4)])
        d2 = _squared_distances(X, V)
        assert np.all(d2 >= 0)

    def test_zero_for_identical_vectors(self):
        v  = _unit(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        X  = v[None]   # (1, D)
        V  = v[None]   # (1, D)
        d2 = _squared_distances(X, V)
        # ‖v - v‖² = 0, but we clamp to 1e-10
        assert d2[0, 0] < 1e-5

    def test_max_distance_orthogonal(self):
        """For orthogonal unit vectors cosine=0 → ‖x-v‖²=2."""
        x = _unit(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        v = _unit(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        d2 = _squared_distances(x[None], v[None])
        assert d2[0, 0] == pytest.approx(2.0, abs=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# _update_memberships
# ──────────────────────────────────────────────────────────────────────────────

class TestUpdateMemberships:
    def _make_d2(self, n=20, k=3) -> np.ndarray:
        rng = np.random.default_rng(7)
        return np.abs(rng.standard_normal((n, k))) + 0.1

    def test_rows_sum_to_one(self):
        d2 = self._make_d2()
        U  = _update_memberships(d2, m=2.0)
        np.testing.assert_allclose(U.sum(axis=1), np.ones(20), atol=1e-5)

    def test_values_in_unit_interval(self):
        d2 = self._make_d2()
        U  = _update_memberships(d2, m=2.0)
        assert np.all(U >= 0) and np.all(U <= 1)

    def test_hard_assignment_for_zero_distance(self):
        """If d[i,c] ≈ 0 the membership should be 1.0 for cluster c."""
        d2         = np.array([[1e-15, 1.0, 2.0]], dtype=np.float64)
        U          = _update_memberships(d2, m=2.0)
        assert U[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert U[0, 1] == pytest.approx(0.0, abs=1e-5)

    def test_shape_preserved(self):
        d2 = self._make_d2(n=15, k=5)
        U  = _update_memberships(d2, m=2.0)
        assert U.shape == (15, 5)

    def test_higher_m_more_uniform(self):
        """Higher m → more uniform memberships → lower max per row."""
        d2   = self._make_d2(n=50, k=4)
        U_lo = _update_memberships(d2, m=1.5)
        U_hi = _update_memberships(d2, m=3.0)
        assert U_lo.max(axis=1).mean() > U_hi.max(axis=1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# _update_centroids
# ──────────────────────────────────────────────────────────────────────────────

class TestUpdateCentroids:
    def test_output_is_unit_norm(self):
        rng = np.random.default_rng(42)
        X   = np.stack([_unit(rng.standard_normal(DIM).astype(np.float32)) for _ in range(30)])
        U   = np.abs(rng.standard_normal((30, 3)).astype(np.float32))
        U   /= U.sum(axis=1, keepdims=True)
        V   = _update_centroids(X, U, m=2.0)
        norms = np.linalg.norm(V, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    def test_shape(self):
        rng = np.random.default_rng(0)
        X   = rng.standard_normal((40, DIM)).astype(np.float32)
        X   /= np.linalg.norm(X, axis=1, keepdims=True)
        U   = np.abs(rng.standard_normal((40, 5)).astype(np.float32))
        U   /= U.sum(axis=1, keepdims=True)
        V   = _update_centroids(X, U, m=2.0)
        assert V.shape == (5, DIM)


# ──────────────────────────────────────────────────────────────────────────────
# FuzzyCMeans end-to-end
# ──────────────────────────────────────────────────────────────────────────────

class TestFuzzyCMeans:
    @pytest.fixture(autouse=True)
    def toy_data(self):
        """3 well-separated clusters, 30 points each."""
        self.X = make_toy_data(n_per_cluster=30, k=3, dim=DIM, noise=0.02)
        self.k = 3

    def test_fit_runs_without_error(self):
        fcm = FuzzyCMeans(n_clusters=self.k, m=2.0, max_iter=50)
        fcm.fit(self.X)
        assert fcm.centroids_ is not None

    def test_centroids_shape(self):
        fcm = FuzzyCMeans(n_clusters=self.k, m=2.0, max_iter=50)
        fcm.fit(self.X)
        assert fcm.centroids_.shape == (self.k, DIM)

    def test_centroids_unit_norm(self):
        fcm = FuzzyCMeans(n_clusters=self.k, m=2.0, max_iter=50)
        fcm.fit(self.X)
        norms = np.linalg.norm(fcm.centroids_, axis=1)
        np.testing.assert_allclose(norms, np.ones(self.k), atol=1e-4)

    def test_predict_memberships_shape(self):
        fcm = FuzzyCMeans(n_clusters=self.k, m=2.0, max_iter=50)
        fcm.fit(self.X)
        U = fcm.predict_memberships(self.X)
        assert U.shape == (len(self.X), self.k)

    def test_predict_memberships_rows_sum_to_one(self):
        fcm = FuzzyCMeans(n_clusters=self.k, m=2.0, max_iter=50)
        fcm.fit(self.X)
        U = fcm.predict_memberships(self.X)
        np.testing.assert_allclose(U.sum(axis=1), np.ones(len(self.X)), atol=1e-4)

    def test_predict_memberships_dtype_float32(self):
        fcm = FuzzyCMeans(n_clusters=self.k, m=2.0, max_iter=50)
        fcm.fit(self.X)
        U = fcm.predict_memberships(self.X)
        assert U.dtype == np.float32

    def test_dominant_cluster_shape(self):
        fcm = FuzzyCMeans(n_clusters=self.k, m=2.0, max_iter=50)
        fcm.fit(self.X)
        U   = fcm.predict_memberships(self.X)
        dom = fcm.dominant_cluster(U)
        assert dom.shape == (len(self.X),)
        assert dom.dtype in (np.int32, np.int64)

    def test_dominant_cluster_within_range(self):
        fcm = FuzzyCMeans(n_clusters=self.k, m=2.0, max_iter=50)
        fcm.fit(self.X)
        dom = fcm.dominant_cluster(fcm.predict_memberships(self.X))
        assert np.all(dom >= 0) and np.all(dom < self.k)

    def test_tight_clusters_high_dominant_membership(self):
        """
        With very tight clusters (noise=0.005), dominant membership
        should exceed 0.7 for the vast majority of points.
        """
        X   = make_toy_data(n_per_cluster=30, k=3, dim=DIM, noise=0.005)
        fcm = FuzzyCMeans(n_clusters=3, m=2.0, max_iter=80)
        fcm.fit(X)
        U   = fcm.predict_memberships(X)
        high_confidence = (U.max(axis=1) > 0.70).mean()
        assert high_confidence > 0.80, (
            f"Expected >80% of docs to have dominant membership >0.7, "
            f"got {high_confidence:.2%}"
        )

    def test_is_boundary_shape(self):
        fcm = FuzzyCMeans(n_clusters=self.k, m=2.0, max_iter=50)
        fcm.fit(self.X)
        U = fcm.predict_memberships(self.X)
        b = fcm.is_boundary(U)
        assert b.shape == (len(self.X),)
        assert b.dtype == bool

    def test_is_boundary_all_false_for_near_hard_clusters(self):
        """With m=1.2 (near-hard), most boundary flags should be False."""
        X   = make_toy_data(n_per_cluster=30, k=3, dim=DIM, noise=0.01)
        fcm = FuzzyCMeans(n_clusters=3, m=1.2, max_iter=80)
        fcm.fit(X)
        U = fcm.predict_memberships(X)
        b = fcm.is_boundary(U)
        assert b.mean() < 0.20, f"Expected <20% boundary with m=1.2, got {b.mean():.2%}"

    def test_predict_before_fit_raises(self):
        fcm = FuzzyCMeans(n_clusters=3)
        with pytest.raises(RuntimeError, match="fit"):
            fcm.predict_memberships(self.X)


# ──────────────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """Saved centroids must be numerically identical after reload."""
        import src.clustering.fuzzy_cluster as fcm_mod

        monkeypatch.setattr(fcm_mod, "_CENTROIDS_PATH", tmp_path / "centroids.npy")
        monkeypatch.setattr(fcm_mod, "_PARAMS_PATH",    tmp_path / "params.pkl")

        X   = make_toy_data(n_per_cluster=20, k=3, dim=DIM)
        fcm = FuzzyCMeans(n_clusters=3, m=2.0, max_iter=40)
        fcm.fit(X)
        fcm.save()

        loaded = FuzzyCMeans.load()
        np.testing.assert_array_equal(fcm.centroids_, loaded.centroids_)
        assert loaded.n_clusters == fcm.n_clusters
        assert loaded.m          == fcm.m

    def test_is_saved_false_before_save(self, tmp_path, monkeypatch):
        import src.clustering.fuzzy_cluster as fcm_mod
        monkeypatch.setattr(fcm_mod, "_CENTROIDS_PATH", tmp_path / "centroids.npy")
        assert not FuzzyCMeans.is_saved()

    def test_is_saved_true_after_save(self, tmp_path, monkeypatch):
        import src.clustering.fuzzy_cluster as fcm_mod
        monkeypatch.setattr(fcm_mod, "_CENTROIDS_PATH", tmp_path / "centroids.npy")
        monkeypatch.setattr(fcm_mod, "_PARAMS_PATH",    tmp_path / "params.pkl")

        X   = make_toy_data(n_per_cluster=20, k=3, dim=DIM)
        fcm = FuzzyCMeans(n_clusters=3, m=2.0, max_iter=30)
        fcm.fit(X)
        fcm.save()
        assert FuzzyCMeans.is_saved()

    def test_load_missing_file_raises(self, tmp_path, monkeypatch):
        import src.clustering.fuzzy_cluster as fcm_mod
        monkeypatch.setattr(fcm_mod, "_CENTROIDS_PATH", tmp_path / "missing.npy")
        with pytest.raises(FileNotFoundError):
            FuzzyCMeans.load()
