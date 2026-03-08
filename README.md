# Trademarkia AI/ML Engineer Task — Semantic Search System

A production-quality semantic search system over the **20 Newsgroups** corpus,
built with three tightly integrated components:

1. **Fuzzy Clustering** — Fuzzy C-Means on sentence embeddings (soft membership per document)
2. **Semantic Cache** — Cluster-bucketed, LRU-evicting cache built from scratch
3. **FastAPI Service** — Live REST API with proper state management

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  POST /query                                                        │
│  ┌──────────────┐   embed    ┌──────────────┐   predict   ┌──────┐ │
│  │ Query string │──────────▶│  Embedder    │───────────▶│ FCM  │ │
│  └──────────────┘           └──────────────┘             └──┬───┘ │
│                                     │                        │     │
│                              q_emb  │           membership   │     │
│                                     ▼                        ▼     │
│                             ┌───────────────────────────────────┐  │
│                             │         SemanticCache             │  │
│                             │  ┌─────────┐  ┌─────────┐        │  │
│                             │  │Bucket 0 │  │Bucket 1 │  ...   │  │
│                             │  └─────────┘  └─────────┘        │  │
│                             │    cosine similarity scan         │  │
│                             └───────┬───────────────────────────┘  │
│                                     │ miss                          │
│                                     ▼                               │
│                             ┌───────────────┐                       │
│                             │  ChromaDB     │  top-5 ANN            │
│                             │  VectorStore  │◀─ filtered by cluster │
│                             └───────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (local, no Docker)

```bash
# 1. Create and activate venv
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file
cp .env.example .env

# 4. Run data pipeline (downloads dataset, embeds, clusters, stores)
#    Takes ~5–15 min on CPU; ~2 min with GPU
python scripts/prepare_data.py

# 5. (Optional) Run cluster quality analysis
python scripts/analyze_clusters.py
# → writes data/cluster_analysis_report.txt

# 6. Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Quick Start (Docker)

```bash
# Build image
docker build -t trademarkia-search .

# Step 1: Run data preparation (once)
docker compose --profile prepare up prepare

# Step 2: Start API
docker compose up api
```

---

## API Reference

### `POST /query`

Embed a natural-language query, check the semantic cache, and return results.

**Request:**
```json
{ "query": "What is the best way to fix a car engine?" }
```

**Response (cache miss):**
```json
{
  "query": "What is the best way to fix a car engine?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "[1] (sim=0.812) Engine trouble shooting...\n\n[2] ...",
  "dominant_cluster": 7
}
```

**Response (cache hit):**
```json
{
  "query": "How do I repair my automobile engine?",
  "cache_hit": true,
  "matched_query": "What is the best way to fix a car engine?",
  "similarity_score": 0.934,
  "result": "[1] (sim=0.812) Engine trouble shooting...",
  "dominant_cluster": 7
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

Flushes all entries and resets counters.

### `GET /health`

Liveness probe returning vector DB doc count and cache state.

---

## Design Decisions

### Part 1 — Data & Embeddings

| Decision | Choice | Rationale |
|---|---|---|
| Preprocessing | Strip headers, footers, quotes (sklearn) | Headers are label leaks; quoted replies fragment meaning |
| Min doc length | 50 chars | Stubs/one-liners carry no topical signal |
| Max doc length | 512 chars | sentence-transformers truncate at ~256 tokens anyway |
| Deduplication | SHA-256 exact hash | Cross-posted duplicates inflate cluster densities |
| Embedding model | `all-MiniLM-L6-v2` | Best speed/quality tradeoff for STS; 384-dim; CPU-friendly |
| Vector DB | ChromaDB | Embedded (no server), persistent, metadata-filtered ANN |

### Part 2 — Fuzzy Clustering

**Why fuzzy?** Hard clusters force an arbitrary label on documents that genuinely
span multiple topics (e.g., gun legislation ∈ politics ∩ firearms).  Fuzzy C-Means
returns a probability distribution over clusters, which is used downstream by the
cache to search multiple buckets for boundary documents.

**Why k=15?** Silhouette analysis (see `data/cluster_analysis_report.txt`) shows
k=15 maximises cluster cohesion.  The 20 original newsgroup labels over-split
technically-adjacent groups (comp.os.ms-windows.misc ≈ comp.windows.x) while the
real latent structure supports ~15 distinct semantic regions.

**The fuzziness exponent m:**

| m | Boundary docs | Interpretation |
|---|---|---|
| 1.2 | ~8% | Near-hard clustering; boundary cases missed |
| **2.0** | **~30%** | **Standard; clear majorities + meaningful uncertainty** |
| 2.5 | ~55% | Over-soft; cluster identity washes out |

m=2.0 is the standard Bezdek default and is used here.  See the analysis script
for a full sweep.

### Part 3 — Semantic Cache

The cache is a **dict-of-lists keyed by dominant cluster**:

```
_buckets = {
    0: [CacheEntry, CacheEntry, ...],   # cluster 0 members
    1: [CacheEntry, ...],               # cluster 1 members
    ...
}
```

**Why cluster-bucketed?**  Naive O(N) scan over all cached entries degrades
as the cache grows.  Because similar queries cluster together, restricting the
scan to the 1–2 relevant buckets reduces average complexity to O(N/k).
With k=15 and 1000 entries, we check ~67 entries per lookup instead of 1000.

Boundary documents (2nd membership > 0.15) are stored in **both** their top-2
cluster buckets so they are findable via either route.

**THE critical parameter — `CACHE_SIMILARITY_THRESHOLD`:**

This controls what counts as "close enough" for a cache hit.

| Threshold | Behaviour | Use case |
|---|---|---|
| 0.99 | Near-exact only | Extremely safety-conscious; very low hit rate |
| **0.90** | **Paraphrases caught; rare false positives** | **Recommended** |
| 0.80 | Topic-similar queries collide | Recall-maximising, higher FP rate |
| 0.70 | Unacceptable false positive rate | Not recommended |

The threshold is exposed via the `CACHE_SIMILARITY_THRESHOLD` environment variable
so it can be tuned in production without a code change.

The `analyze_clusters.py` script sweeps thresholds over hand-labelled
same-intent / different-intent query pairs and prints a precision/recall table.

---

## Project Structure

```
trademarkia_task/
├── src/
│   ├── config.py               # All tuneable hyper-parameters
│   ├── data/
│   │   └── loader.py           # Corpus loading + cleaning
│   ├── embeddings/
│   │   └── embedder.py         # sentence-transformers wrapper
│   ├── vectordb/
│   │   └── store.py            # ChromaDB wrapper
│   ├── clustering/
│   │   └── fuzzy_cluster.py    # Fuzzy C-Means from scratch
│   ├── cache/
│   │   └── semantic_cache.py   # Semantic cache from scratch
│   └── api/
│       ├── main.py             # FastAPI app (lifespan, endpoints)
│       └── models.py           # Pydantic schemas
├── scripts/
│   ├── prepare_data.py         # One-off data pipeline
│   └── analyze_clusters.py     # Quality analysis + threshold sweep
├── data/                       # Created at runtime
│   ├── chroma_db/              # Persistent vector store
│   └── clusters/               # FCM centroids + params
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```
