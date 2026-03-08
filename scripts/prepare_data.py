
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.loader import load_corpus, category_names
from src.embeddings.embedder import Embedder
from src.clustering.fuzzy_cluster import FuzzyCMeans
from src.vectordb.store import VectorStore
from src.config import N_CLUSTERS, FCM_M


def main() -> None:
    t0 = time.time()

    print("\n═══ Step 1 / 5 · Load corpus ═══")
    texts, labels, ids = load_corpus(subset="all")
    cat_names = category_names()
    print(f"Categories: {cat_names}")

    print("\n═══ Step 2 / 5 · Embed corpus ═══")
    embedder = Embedder()
    embeddings = embedder.embed(texts, batch_size=128, show_progress=True)
    print(f"Embeddings shape: {embeddings.shape}")

    print(f"\n═══ Step 3 / 5 · Fit Fuzzy C-Means (k={N_CLUSTERS}, m={FCM_M}) ═══")
    fcm = FuzzyCMeans(n_clusters=N_CLUSTERS, m=FCM_M)
    fcm.fit(embeddings)
    fcm.save()

    print("\n═══ Step 4 / 5 · Assign memberships ═══")
    BATCH = 1024
    all_memberships = []
    for i in range(0, len(embeddings), BATCH):
        batch = embeddings[i : i + BATCH]
        all_memberships.append(fcm.predict_memberships(batch))
    memberships = np.vstack(all_memberships)          # (N, k)
    dominant    = fcm.dominant_cluster(memberships)   # (N,)
    boundary    = fcm.is_boundary(memberships)        # (N,) bool

    print(f"  Dominant cluster distribution:")
    for c in range(N_CLUSTERS):
        count = int((dominant == c).sum())
        bar   = "█" * (count // 100)
        print(f"  Cluster {c:2d}: {count:4d} docs {bar}")

    print(f"\n  Boundary docs (2nd membership > 0.15): {boundary.sum():,} "
          f"({100*boundary.mean():.1f} %)")

    print("\n═══ Step 5 / 5 · Upsert to ChromaDB ═══")
    store = VectorStore()

    metadatas = [
        {
            "label":            labels[i],
            "category":         cat_names[labels[i]],
            "dominant_cluster": int(dominant[i]),
            "is_boundary":      bool(boundary[i]),
            "mem_top1":         float(round(float(np.sort(memberships[i])[-1]), 4)),
            "mem_top2":         float(round(float(np.sort(memberships[i])[-2]), 4)),
        }
        for i in range(len(texts))
    ]

    UPSERT_BATCH = 512
    for start in range(0, len(texts), UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, len(texts))
        store.upsert(
            ids        = ids[start:end],
            embeddings = embeddings[start:end],
            documents  = texts[start:end],
            metadatas  = metadatas[start:end],
        )
        print(f"  Upserted {end}/{len(texts)}", end="\r")

    print(f"\n  ChromaDB now contains {store.count():,} documents.")

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.1f} s")
    print("  Start the API with:")
    print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()
