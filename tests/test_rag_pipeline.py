"""
RAG Pipeline Evaluation Tests
==============================
Tests retrieval quality and pipeline integrity WITHOUT calling the LLM
(fast, free, no API key needed for tests).

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short    # shorter tracebacks
"""

import sys
import os
import pytest
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vectorstore():
    """
    Initialize the vectorstore fixture for the test session.

    Returns:
        Chroma: The persistent vectorstore instance.
    """
    db_dir = ROOT_DIR / "data" / "chroma_db"
    if not db_dir.exists() or not any(db_dir.iterdir()):
        pytest.skip("data/chroma_db not built yet — run the app once to build the index.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=str(db_dir), embedding_function=embeddings)


# ---------------------------------------------------------------------------
# 1. Index Integrity Tests
# ---------------------------------------------------------------------------

class TestIndexIntegrity:

    def test_chroma_db_exists(self):
        """The vector database directory must exist and be non-empty."""
        db_dir = ROOT_DIR / "data" / "chroma_db"
        assert db_dir.exists(), "data/chroma_db/ directory does not exist. Run the app to build it."
        assert any(db_dir.iterdir()), "data/chroma_db/ is empty. Index was not built correctly."

    def test_data_directory_has_transcripts(self):
        """The data/ directory must contain at least one transcript file."""
        data_dir = ROOT_DIR / "data"
        assert data_dir.exists(), "data/ directory does not exist."
        all_files = list(data_dir.rglob("*.txt")) + list(data_dir.rglob("*.pdf")) + list(data_dir.rglob("*.json"))
        assert len(all_files) > 0, "No transcript files found in data/. Add .txt, .pdf or .json files."

    def test_vectorstore_loads(self, vectorstore):
        """ChromaDB must load without errors."""
        assert vectorstore is not None

    def test_vectorstore_has_documents(self, vectorstore):
        """The vectorstore must contain indexed documents."""
        count = vectorstore._collection.count()
        assert count > 0, f"Vector store is empty (0 documents). Re-build the index."
        print(f"\n  ✅ Vector store contains {count} chunks")


# ---------------------------------------------------------------------------
# 2. Retrieval Quality Tests
# ---------------------------------------------------------------------------
#
# Each test checks that a trading question retrieves AT LEAST ONE relevant
# chunk — judged by keyword presence in the returned content.
# These questions are grounded in Mark Minervini's momentum/price-action
# teaching style based on the transcript content.
#

RETRIEVAL_CASES = [
    {
        "id": "batting_average",
        "query": "What is a good batting average for a trader?",
        "expected_keywords": ["batting average", "win", "loss", "percent"],
        "description": "Batting average concept must be retrievable",
    },
    {
        "id": "stop_loss",
        "query": "How should I set my stop loss?",
        "expected_keywords": ["stop", "loss", "risk", "percent"],
        "description": "Stop loss rules must be retrievable",
    },
    {
        "id": "breakout",
        "query": "How do I trade a breakout?",
        "expected_keywords": ["breakout", "volume", "price", "momentum"],
        "description": "Breakout trading criteria must be retrievable",
    },
    {
        "id": "position_sizing",
        "query": "How do I size my positions correctly?",
        "expected_keywords": ["position", "risk", "capital", "size"],
        "description": "Position sizing guidance must be retrievable",
    },
    {
        "id": "risk_reward",
        "query": "What risk to reward ratio should I aim for?",
        "expected_keywords": ["risk", "reward", "ratio", "gain", "loss"],
        "description": "Risk/reward concepts must be retrievable",
    },
]


class TestRetrievalQuality:

    @pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=[c["id"] for c in RETRIEVAL_CASES])
    def test_retrieves_relevant_chunks(self, vectorstore, case):
        """
        For each known trading question, at least one retrieved chunk
        must contain at least one of the expected keywords.
        """
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
            case["query"], k=6
        )
        assert len(docs_with_scores) > 0, f"No documents retrieved for: '{case['query']}'"

        # Combine all retrieved text for keyword matching
        all_text = " ".join(doc.page_content.lower() for doc, _ in docs_with_scores)
        matched_keywords = [kw for kw in case["expected_keywords"] if kw in all_text]

        assert len(matched_keywords) > 0, (
            f"[{case['id']}] Expected at least one of {case['expected_keywords']} "
            f"in retrieved chunks, but found none.\n"
            f"Retrieved text preview: {all_text[:400]}"
        )
        print(f"\n  ✅ '{case['id']}': matched keywords {matched_keywords}")

    @pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=[c["id"] for c in RETRIEVAL_CASES])
    def test_relevance_scores_are_nonzero(self, vectorstore, case):
        """
        Relevance scores for known trading questions must be > 0.1
        (ChromaDB cosine relevance: 0 = worst, 1 = perfect match).
        """
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
            case["query"], k=3
        )
        top_score = max(score for _, score in docs_with_scores)
        assert top_score > 0.1, (
            f"[{case['id']}] Top relevance score {top_score:.3f} is too low — "
            f"the index may not contain relevant content for this query."
        )
        print(f"\n  ✅ '{case['id']}': top relevance score = {top_score:.3f}")

    def test_irrelevant_query_scores_lower(self, vectorstore):
        """
        A completely off-topic query should score lower than a
        trading-relevant query (basic sanity check on embedding quality).
        """
        trading_query = "How do I cut my losses and let my winners run?"
        irrelevant_query = "What is the recipe for chocolate cake?"

        trading_docs = vectorstore.similarity_search_with_relevance_scores(trading_query, k=3)
        irrelevant_docs = vectorstore.similarity_search_with_relevance_scores(irrelevant_query, k=3)

        trading_top = max(score for _, score in trading_docs)
        irrelevant_top = max(score for _, score in irrelevant_docs)

        print(f"\n  Trading query score:    {trading_top:.3f}")
        print(f"  Irrelevant query score: {irrelevant_top:.3f}")

        assert trading_top > irrelevant_top, (
            f"Expected trading query ({trading_top:.3f}) to score higher than "
            f"irrelevant query ({irrelevant_top:.3f}). Embeddings may not be working correctly."
        )


# ---------------------------------------------------------------------------
# 3. Prompt Template Tests
# ---------------------------------------------------------------------------

class TestPromptTemplate:

    def test_prompt_has_context_placeholder(self):
        """The system prompt must contain {context} for RAG injection."""
        from src.prompts import TRADING_MENTOR_PROMPT
        assert "{context}" in TRADING_MENTOR_PROMPT, "TRADING_MENTOR_PROMPT missing {context} placeholder"

    def test_prompt_has_question_placeholder(self):
        """The system prompt must contain {question} for user query injection."""
        from src.prompts import TRADING_MENTOR_PROMPT
        assert "{question}" in TRADING_MENTOR_PROMPT, "TRADING_MENTOR_PROMPT missing {question} placeholder"

    def test_prompt_enforces_rag_rules(self):
        """The prompt should instruct the model to stay within transcript content."""
        from src.prompts import TRADING_MENTOR_PROMPT
        # Basic sanity: the prompt should contain grounding instructions
        prompt_lower = TRADING_MENTOR_PROMPT.lower()
        assert any(kw in prompt_lower for kw in ["only", "excerpts", "transcript", "notes"]), (
            "Prompt does not appear to contain RAG grounding instructions."
        )
