
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm

from src.config import (
    N_CLUSTERS, FCM_M, FCM_MAX_ITER, FCM_TOL,
    CLUSTER_SAMPLE, CLUSTER_DIR, EMBEDDING_DIM,
)

_CENTROIDS_PATH   = CLUSTER_DIR / "centroids.npy"
_PARAMS_PATH      = CLUSTER_DIR / "params.pkl"


def _squared_distances(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute ‖x_i - v_c‖² for all i,c using the cosine-space trick.

    X: (N, D) unit-norm embeddings
    V: (k, D) unit-norm centroids
    Returns: (N, k) non-negative distances
    """
    dots = X @ V.T                         
    d2 = np.maximum(2.0 - 2.0 * dots, 1e-10)  
    return d2.astype(np.float64)


def _update_memberships(d2: np.ndarray, m: float) -> np.ndarray:
    """
    FCM membership update.

    d2 : (N, k) squared distances
    m  : fuzziness exponent
    Returns: (N, k) membership matrix U, rows sum to 1
    """
    # exponent for the ratio: 2 / (m - 1)
    exp = 2.0 / (m - 1.0)
    
    U = np.zeros_like(d2)
    for c in range(d2.shape[1]):
        ratio = (d2[:, c:c+1] / d2) ** exp   # (N, k)
        U[:, c] = 1.0 / ratio.sum(axis=1)
    exact = np.any(d2 < 1e-10, axis=1)
    if exact.any():
        U[exact] = 0.0
        U[exact, np.argmin(d2[exact], axis=1)] = 1.0
    return U


def _update_centroids(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    """
    FCM centroid update.

    X : (N, D)
    U : (N, k)
    Returns: (k, D) re-normalised centroids
    """
    Um = U ** m                        # (N, k)
    V = (Um.T @ X) / Um.sum(axis=0)[:, None]   # (k, D)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.maximum(norms, 1e-8)
    return V.astype(np.float32)



class FuzzyCMeans:
    """Fuzzy C-Means fitted on a sample, applied to the full corpus."""

    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        m: float = FCM_M,
        max_iter: int = FCM_MAX_ITER,
        tol: float = FCM_TOL,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.centroids_: Optional[np.ndarray] = None   # (k, D) float32


    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        """
        Fit FCM centroids on X (N, D) unit-norm embeddings.
        If N > CLUSTER_SAMPLE, a random subsample is used for speed.
        """
        N = X.shape[0]
        if N > CLUSTER_SAMPLE:
            idx = self.rng.choice(N, CLUSTER_SAMPLE, replace=False)
            Xs = X[idx]
        else:
            Xs = X

        k, D = self.n_clusters, Xs.shape[1]

        init_idx = self.rng.choice(len(Xs), k, replace=False)
        V = Xs[init_idx].astype(np.float64)

        prev_J = np.inf
        print(f"[FCM] Fitting k={k}, m={self.m}, n_sample={len(Xs)}")
        for iteration in tqdm(range(self.max_iter), desc="FCM iter"):
            d2 = _squared_distances(Xs, V)
            U  = _update_memberships(d2, self.m)
            V_new = _update_centroids(Xs, U, self.m)

            J = float(np.sum((U ** self.m) * d2))
            delta = abs(prev_J - J)
            prev_J = J

            V = V_new.astype(np.float64)
            if delta < self.tol:
                print(f"[FCM] Converged at iteration {iteration+1}, J={J:.6f}")
                break
        else:
            print(f"[FCM] Reached max_iter={self.max_iter}, J={J:.6f}")

        self.centroids_ = V.astype(np.float32)
        return self


    def predict_memberships(self, X: np.ndarray) -> np.ndarray:
        """
        Compute membership matrix for X.

        Returns
        -------
        U : (N, k) float32, rows sum to 1.
        """
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict_memberships().")
        d2 = _squared_distances(X, self.centroids_)
        U  = _update_memberships(d2, self.m)
        return U.astype(np.float32)

    def dominant_cluster(self, U: np.ndarray) -> np.ndarray:
        """Return argmax cluster per document.  Shape: (N,)."""
        return np.argmax(U, axis=1).astype(np.int32)

    def is_boundary(self, U: np.ndarray, threshold: float = 0.15) -> np.ndarray:
        """
        Boolean mask: True if a doc's second-best membership > threshold.
        Boundary documents reveal semantic overlap between clusters and are
        the most interesting cases for cache lookup (search top-2 clusters).
        """
        sorted_U = np.sort(U, axis=1)[:, ::-1]
        return (sorted_U[:, 1] > threshold).astype(bool)


    def save(self) -> None:
        np.save(_CENTROIDS_PATH, self.centroids_)
        with open(_PARAMS_PATH, "wb") as f:
            pickle.dump({"n_clusters": self.n_clusters, "m": self.m}, f)
        print(f"[FCM] Saved centroids to {_CENTROIDS_PATH}")

    @classmethod
    def load(cls) -> "FuzzyCMeans":
        if not _CENTROIDS_PATH.exists():
            raise FileNotFoundError(
                "No saved FCM model found.  Run `python scripts/prepare_data.py` first."
            )
        with open(_PARAMS_PATH, "rb") as f:
            params = pickle.load(f)
        model = cls(n_clusters=params["n_clusters"], m=params["m"])
        model.centroids_ = np.load(_CENTROIDS_PATH)
        print(f"[FCM] Loaded centroids: k={model.n_clusters}, m={model.m}")
        return model

    @classmethod
    def is_saved(cls) -> bool:
        return _CENTROIDS_PATH.exists()



def select_k(
    X: np.ndarray,
    k_range: range = range(10, 26),
    m: float = 2.0,
) -> dict:
    """
    Sweep k values and return silhouette + Davies-Bouldin scores.
    Used in analyze_clusters.py to justify N_CLUSTERS=15.
    """
    results = {}
    for k in k_range:
        fcm = FuzzyCMeans(n_clusters=k, m=m, max_iter=100)
        fcm.fit(X)
        U = fcm.predict_memberships(X[:CLUSTER_SAMPLE])
        labels = fcm.dominant_cluster(U)
        Xs = X[:CLUSTER_SAMPLE] if len(X) > CLUSTER_SAMPLE else X

        try:
            sil = silhouette_score(Xs, labels, metric="cosine", sample_size=2000)
            db  = davies_bouldin_score(Xs, labels)
        except ValueError:
            sil, db = 0.0, 999.0

        results[k] = {"silhouette": sil, "davies_bouldin": db}
        print(f"  k={k:2d}  silhouette={sil:.4f}  davies_bouldin={db:.4f}")

    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"\n[select_k] Best k by silhouette: {best_k}")
    return results
