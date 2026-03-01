"""
retrieval.py
------------
Two-stage advanced retrieval pipeline:

Stage 1 — Hybrid Search:  Combines BM25 keyword matching with vector similarity
                          by merging results from both retrievers.
Stage 2 — Cross-Encoder Re-ranking:  Re-scores the merged results using
                          cross-encoder/ms-marco-MiniLM-L-6-v2 for higher precision.

Usage in app.py:
    from retrieval import build_bm25_retriever, hybrid_retrieve_and_rerank

    bm25 = build_bm25_retriever(vectorstore)
    docs, scores = hybrid_retrieve_and_rerank(query, vectorstore, bm25, k=6)
"""

import logging
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ── Cross-encoder (loaded once, cached at module level) ──────────────────────
_cross_encoder = None


def _get_cross_encoder() -> CrossEncoder:
    """Lazy-load the cross-encoder model (first call downloads ~90 MB)."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model for re-ranking...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


# ── Stage 1: Hybrid Retrieval ────────────────────────────────────────────────

def build_bm25_retriever(vectorstore, k: int = 12) -> BM25Retriever:
    """
    Construct a BM25Retriever by indexing all document text currently in the vectorstore.

    This allows for effective keyword-based retrieval alongside semantic search.

    Args:
        vectorstore (Chroma): The ChromaDB vectorstore instance containing documents.
        k (int): Number of documents to retrieve initially. Defaults to 12.

    Returns:
        BM25Retriever: A configured BM25 retriever, or None if the vectorstore is empty.
    """
    # Get all documents from the vectorstore
    all_docs_data = vectorstore.get(include=["documents", "metadatas"])
    documents = all_docs_data.get("documents", [])
    metadatas = all_docs_data.get("metadatas", [])

    if not documents:
        logger.warning("No documents found in vectorstore for BM25 indexing.")
        return None

    # Convert to LangChain Documents for BM25Retriever
    from langchain_core.documents import Document
    langchain_docs = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(documents, metadatas)
    ]

    bm25_retriever = BM25Retriever.from_documents(langchain_docs, k=k)
    logger.info(f"BM25 retriever built with {len(langchain_docs)} documents.")
    return bm25_retriever


def hybrid_retrieve(
    query: str,
    vectorstore,
    bm25_retriever: BM25Retriever | None,
    k: int = 6,
) -> list:
    """
    Stage 1: Retrieve chunks using an interleaved merge of BM25 and vector search.

    Args:
        query (str): The search query.
        vectorstore (Chroma): The vector database.
        bm25_retriever (BM25Retriever): Pre-indexed keyword retriever.
        k (int): Number of final unique chunks to target (before re-ranking).

    Returns:
        list: A list of (Document, score) tuples.
    """
    if bm25_retriever is None:
        # Fallback to pure vector search
        return vectorstore.similarity_search_with_relevance_scores(query, k=k)

    # Over-retrieve to give the re-ranker more candidates
    fetch_k = k * 2

    # 1. Get BM25 results
    bm25_retriever.k = fetch_k
    bm25_docs = bm25_retriever.invoke(query)

    # 2. Get Vector results with scores
    vector_results = vectorstore.similarity_search_with_relevance_scores(query, k=fetch_k)
    vector_docs = [doc for doc, _ in vector_results]
    score_lookup = {hash(doc.page_content): score for doc, score in vector_results}

    # 3. Interleave results evenly (1 from vector, 1 from BM25, etc.) until deduplicated fetch_k is reached
    merged = []
    seen = set()
    v_idx, b_idx = 0, 0
    
    while len(merged) < fetch_k and (v_idx < len(vector_docs) or b_idx < len(bm25_docs)):
        # Add from vector
        while v_idx < len(vector_docs):
            doc = vector_docs[v_idx]
            v_idx += 1
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                merged.append((doc, score_lookup.get(h, 0.0)))
                break
                
        # Add from BM25
        if len(merged) >= fetch_k: break
        
        while b_idx < len(bm25_docs):
            doc = bm25_docs[b_idx]
            b_idx += 1
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                merged.append((doc, score_lookup.get(h, 0.0))) # Fast fallback for missing vector scores
                break

    logger.info(f"Hybrid retrieval: {len(merged)} unique chunks from BM25 + vector merge.")
    return merged


# ── Stage 2: Cross-Encoder Re-ranking ────────────────────────────────────────

def rerank_documents(
    query: str,
    doc_score_pairs: list,
    top_k: int = 6,
) -> list:
    """
    Stage 2: Re-rank retrieval candidates using a Cross-Encoder for precision.

    Args:
        query (str): The search query.
        doc_score_pairs (list): List of (Document, score) from Stage 1.
        top_k (int): Number of final documents to return.

    Returns:
        list: (Document, rerank_score) tuples, sorted descending by score.
    """
    if not doc_score_pairs:
        return []

    cross_encoder = _get_cross_encoder()

    # Build query-document pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc, _ in doc_score_pairs]

    # Score all pairs
    ce_scores = cross_encoder.predict(pairs)

    # Combine with original docs and sort by cross-encoder score (descending)
    scored = [
        (doc, float(ce_score))
        for (doc, _), ce_score in zip(doc_score_pairs, ce_scores)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_results = scored[:top_k]
    logger.info(
        f"Re-ranked {len(doc_score_pairs)} → top {len(top_results)} chunks. "
        f"Best score: {top_results[0][1]:.3f}, Worst: {top_results[-1][1]:.3f}"
    )
    return top_results


# ── Combined Pipeline ────────────────────────────────────────────────────────

def hybrid_retrieve_and_rerank(
    query: str,
    vectorstore,
    bm25_retriever: BM25Retriever | None,
    k: int = 6,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> list:
    """
    Full two-stage retrieval pipeline.

    Args:
        query:           Search query.
        vectorstore:     ChromaDB vectorstore instance.
        bm25_retriever:  Pre-built BM25Retriever (or None for vector-only).
        k:               Number of chunks to return.
        use_hybrid:      If False, skip BM25 and use only vector search.
        use_rerank:      If False, skip cross-encoder re-ranking.

    Returns:
        List of (Document, score) tuples, length ≤ k.
    """
    # Stage 1: Retrieve
    if use_hybrid and bm25_retriever is not None:
        candidates = hybrid_retrieve(query, vectorstore, bm25_retriever, k=k)
    else:
        candidates = vectorstore.similarity_search_with_relevance_scores(query, k=k if not use_rerank else k * 2)

    # Stage 2: Re-rank
    if use_rerank and len(candidates) > 0:
        results = rerank_documents(query, candidates, top_k=k)
    else:
        results = candidates[:k]

    return results
