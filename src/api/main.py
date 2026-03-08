
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.embeddings.embedder import Embedder
from src.clustering.fuzzy_cluster import FuzzyCMeans
from src.vectordb.store import VectorStore
from src.cache.semantic_cache import SemanticCache
from src.api.models import QueryRequest, QueryResponse, CacheStats, FlushResponse
from src.config import CACHE_SIMILARITY_THRESHOLD



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once at startup."""
    print("[startup] Loading embedder …")
    app.state.embedder = Embedder()
    app.state.embedder.embed_one("warm up")   
    print("[startup] Loading FCM model …")
    if not FuzzyCMeans.is_saved():
        raise RuntimeError(
            "FCM model not found on disk.  "
            "Run `python scripts/prepare_data.py` before starting the API."
        )
    app.state.fcm = FuzzyCMeans.load()

    print("[startup] Opening vector store …")
    app.state.store = VectorStore()
    if not app.state.store.is_populated():
        raise RuntimeError(
            "ChromaDB is empty.  "
            "Run `python scripts/prepare_data.py` before starting the API."
        )

    print("[startup] Initialising semantic cache …")
    app.state.cache = SemanticCache(threshold=CACHE_SIMILARITY_THRESHOLD)

    print(f"[startup] Ready.  VectorDB has {app.state.store.count():,} documents.")
    yield
    print("[shutdown] Goodbye.")



app = FastAPI(
    title="Trademarkia Semantic Search",
    description=(
        "Fuzzy-clustered semantic search with an LRU-evicting, "
        "cluster-bucketed semantic cache over the 20 Newsgroups corpus."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute a result for a query (cache miss path)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_result(
    query:     str,
    q_emb:     Any,
    dominant:  int,
    store:     VectorStore,
) -> str:
    """
    Retrieve the top-5 documents from the vector store and format a result.

    In a production system this would call an LLM.  Here we return a
    structured snippet of the nearest-neighbour documents to keep the
    service self-contained and API-key-free.
    """
    results = store.query(
        query_embedding=q_emb,
        n_results=5,
        where={"dominant_cluster": {"$eq": dominant}},
    )

    if not results["documents"] or not results["documents"][0]:
        results = store.query(query_embedding=q_emb, n_results=5)

    docs = results["documents"][0]
    distances = results["distances"][0]

    snippets = []
    for i, (doc, dist) in enumerate(zip(docs, distances), 1):
        similarity = round(1.0 - dist, 4)   
        snippets.append(f"[{i}] (sim={similarity:.3f}) {doc[:200]}")

    return "\n\n".join(snippets) if snippets else "No relevant documents found."


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Embed the query, check the semantic cache, and return the result.

    On a cache miss: compute the result via vector-store retrieval,
    store in cache, then return.
    """
    embedder: Embedder      = app.state.embedder
    fcm:      FuzzyCMeans   = app.state.fcm
    store:    VectorStore   = app.state.store
    cache:    SemanticCache = app.state.cache

    q_emb = embedder.embed_one(request.query)          

    membership = fcm.predict_memberships(q_emb[None])[0]  
    dominant   = int(membership.argmax())

    hit = cache.lookup(q_emb, membership)

    if hit is not None:
        entry, sim = hit
        return QueryResponse(
            query            = request.query,
            cache_hit        = True,
            matched_query    = entry.original_query,
            similarity_score = round(float(sim), 4),
            result           = entry.result,
            dominant_cluster = entry.dominant_cluster,
        )

    result = _compute_result(request.query, q_emb, dominant, store)

    cache.store(
        query_embedding = q_emb,
        original_query  = request.query,
        result          = result,
        membership      = membership,
    )

    return QueryResponse(
        query            = request.query,
        cache_hit        = False,
        matched_query    = None,
        similarity_score = None,
        result           = result,
        dominant_cluster = dominant,
    )


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats() -> CacheStats:
    """Return current cache hit/miss statistics."""
    return CacheStats(**app.state.cache.stats)


@app.delete("/cache", response_model=FlushResponse)
async def flush_cache() -> FlushResponse:
    """Flush the cache entirely and reset all stats."""
    cache: SemanticCache = app.state.cache
    entries_before = cache.stats["total_entries"]
    cache.flush()
    return FlushResponse(
        message         = "Cache flushed successfully.",
        entries_cleared = entries_before,
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness probe."""
    return {
        "status":          "ok",
        "vector_db_docs":  app.state.store.count(),
        "cache_entries":   app.state.cache.stats["total_entries"],
        "cache_threshold": app.state.cache.threshold,
    }
