# Advanced Retrieval Strategy

## The Problem with Vanilla Vector Search

Pure vector similarity search has two weaknesses:

1. **Semantic drift** — The embedding model captures meaning but can miss exact keyword matches. A query for "200-day moving average" might retrieve chunks about "long-term trend indicators" but miss the chunk that literally says "200-day".

2. **Noisy ranking** — Bi-encoder models (like all-MiniLM-L6-v2) embed the query and document independently. They can't jointly evaluate whether a specific chunk actually answers the query — they just measure semantic proximity.

## Our Two-Stage Solution

### Stage 1: Hybrid Search (BM25 + Vector)

| Method                      | Strength                                 | Weakness                            |
| --------------------------- | ---------------------------------------- | ----------------------------------- |
| **BM25** (keyword)          | Exact term matching, no hallucination    | Misses synonyms and paraphrases     |
| **Vector** (semantic)       | Understands meaning, handles paraphrases | Can miss exact terms                |
| **Hybrid** (50/50 ensemble) | Gets the best of both                    | Slightly more candidates to process |

We use LangChain's `EnsembleRetriever` with equal weights (0.5 BM25, 0.5 vector). The BM25 index is built from the same chunks stored in ChromaDB, so there's no additional data to manage.

**Over-retrieval**: We fetch `k × 2` candidates to give the re-ranker more material to work with.

### Stage 2: Cross-Encoder Re-ranking

After hybrid retrieval, we re-score every candidate using `cross-encoder/ms-marco-MiniLM-L-6-v2`:

| Approach                    | How it works                                                       | Quality                       |
| --------------------------- | ------------------------------------------------------------------ | ----------------------------- |
| **Bi-encoder** (Stage 1)    | Embeds query and doc separately, compares vectors                  | Fast but approximate          |
| **Cross-encoder** (Stage 2) | Feeds query + doc together into one model, outputs relevance score | Slower but much more accurate |

The cross-encoder jointly attends to the query and the chunk, catching subtle relevance signals that bi-encoders miss. This is the same approach used by Google Search and Bing.

**Trade-off**: Cross-encoders are ~10× slower than bi-encoders, which is why we only run them on the pre-filtered candidates (typically 12 chunks), not the full corpus.

## Why These Two?

| Considered                    | Chosen? | Reason                                                      |
| ----------------------------- | ------- | ----------------------------------------------------------- |
| Hybrid search (BM25 + vector) | ✅      | Biggest recall improvement for exact trading terms          |
| Cross-encoder re-ranking      | ✅      | Biggest precision improvement, industry standard            |
| Query expansion               | ❌      | Already have query reformulation for conversational context |
| Metadata filtering            | ❌      | Less impactful for this corpus size; could add later        |

## Model Details

- **BM25**: `rank_bm25` library (Okapi BM25 algorithm)
- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~90MB, runs locally on CPU/MPS)
- **Bi-encoder**: `all-MiniLM-L6-v2` (same as our embedding model)

Both run locally — no API calls, no cost.

## Sidebar Controls

Users can toggle each stage independently:

- **🔀 Hybrid Search** — ON/OFF (default: ON)
- **📊 Cross-Encoder Re-ranking** — ON/OFF (default: ON)

Turning both off falls back to vanilla vector similarity search.

---

_Last updated: 2026-03-01_
