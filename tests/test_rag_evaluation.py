"""
RAG Evaluation Tests
=====================
Lightweight tests that validate the golden QA dataset and measure
retrieval recall WITHOUT calling the LLM (fast, free, no API key).

Run with:
    pytest tests/test_rag_evaluation.py -v
"""

import sys
import json
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

GOLDEN_QA_PATH = ROOT_DIR / "tests" / "golden_qa.json"
REQUIRED_FIELDS = {"id", "question", "expected_answer_keywords", "expected_context_keywords", "category"}
VALID_CATEGORIES = {"risk_management", "entries", "fundamentals", "vcp", "mindset"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def golden_qa():
    """
    Load the golden QA dataset for testing.

    Returns:
        list[dict]: The parsed golden QA entries.
    """
    assert GOLDEN_QA_PATH.exists(), f"Golden QA file not found: {GOLDEN_QA_PATH}"
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0, "Golden QA dataset is empty"
    return data


@pytest.fixture(scope="session")
def vectorstore():
    """
    Initialize the vectorstore fixture for the session.

    Returns:
        Chroma: The persistent vectorstore instance.
    """
    db_dir = ROOT_DIR / "data" / "chroma_db"
    if not db_dir.exists() or not any(db_dir.iterdir()):
        pytest.skip("data/chroma_db not built yet — run the app once to build the index.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=str(db_dir), embedding_function=embeddings)


# ---------------------------------------------------------------------------
# 1. Golden Dataset Validation
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    def test_file_exists(self):
        """The golden QA file must exist."""
        assert GOLDEN_QA_PATH.exists(), f"Missing: {GOLDEN_QA_PATH}"

    def test_valid_json(self, golden_qa):
        """The file must contain a valid JSON array."""
        assert isinstance(golden_qa, list)

    def test_minimum_entries(self, golden_qa):
        """Must have at least 10 evaluation entries."""
        assert len(golden_qa) >= 10, f"Only {len(golden_qa)} entries; need at least 10"

    def test_required_fields(self, golden_qa):
        """Every entry must contain all required fields."""
        for i, entry in enumerate(golden_qa):
            missing = REQUIRED_FIELDS - set(entry.keys())
            assert not missing, f"Entry {i} ('{entry.get('id', '?')}') missing fields: {missing}"

    def test_no_duplicate_ids(self, golden_qa):
        """All entry IDs must be unique."""
        ids = [e["id"] for e in golden_qa]
        duplicates = [x for x in ids if ids.count(x) > 1]
        assert not duplicates, f"Duplicate IDs: {set(duplicates)}"

    def test_valid_categories(self, golden_qa):
        """All categories must be from the allowed set."""
        for entry in golden_qa:
            assert entry["category"] in VALID_CATEGORIES, (
                f"Entry '{entry['id']}' has invalid category '{entry['category']}'. "
                f"Allowed: {VALID_CATEGORIES}"
            )

    def test_keywords_are_nonempty(self, golden_qa):
        """Expected keywords lists must not be empty."""
        for entry in golden_qa:
            assert len(entry["expected_answer_keywords"]) > 0, (
                f"Entry '{entry['id']}' has empty expected_answer_keywords"
            )
            assert len(entry["expected_context_keywords"]) > 0, (
                f"Entry '{entry['id']}' has empty expected_context_keywords"
            )


# ---------------------------------------------------------------------------
# 2. Retrieval Recall Tests
# ---------------------------------------------------------------------------

class TestRetrievalRecall:

    @pytest.fixture(scope="class")
    def recall_scores(self, golden_qa, vectorstore):
        """
        Compute retrieval recall for each query in the golden dataset.

        Args:
            golden_qa (list): The evaluative dataset.
            vectorstore (Chroma): The vector database.

        Returns:
            list[dict]: Scores containing recall% and matched/missed keywords.
        """
        scores = []
        for qa in golden_qa:
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
                qa["question"], k=6
            )
            retrieved_text = " ".join(
                doc.page_content.lower() for doc, _ in docs_with_scores
            )
            expected = qa["expected_context_keywords"]
            hits = sum(1 for kw in expected if kw.lower() in retrieved_text)
            recall = hits / len(expected) if expected else 0.0
            scores.append({
                "id": qa["id"],
                "recall": recall,
                "matched": [kw for kw in expected if kw.lower() in retrieved_text],
                "missed": [kw for kw in expected if kw.lower() not in retrieved_text],
            })
        return scores

    @pytest.mark.parametrize(
        "qa_index",
        range(12),
        ids=[
            "vcp_definition", "200_day_ma_rule", "stop_loss_rule",
            "market_leader_characteristics", "line_of_least_resistance",
            "position_sizing", "breakout_volume", "trading_discipline_mindset",
            "relative_strength_threshold", "primary_base",
            "cyclical_stocks_avoidance", "earnings_acceleration",
        ],
    )
    def test_individual_retrieval_recall(self, recall_scores, qa_index):
        """Each question must retrieve at least one expected context keyword."""
        if qa_index >= len(recall_scores):
            pytest.skip("Golden QA has fewer entries than expected")
        score = recall_scores[qa_index]
        assert score["recall"] > 0, (
            f"[{score['id']}] No expected keywords found in retrieved chunks. "
            f"Missed: {score['missed']}"
        )
        print(f"\n  ✅ '{score['id']}': recall={score['recall']:.0%}, matched={score['matched']}")

    def test_aggregate_retrieval_recall(self, recall_scores):
        """Mean retrieval recall across all questions must be ≥ 50%."""
        mean_recall = sum(s["recall"] for s in recall_scores) / len(recall_scores)
        print(f"\n  📊 Aggregate retrieval recall: {mean_recall:.0%}")
        assert mean_recall >= 0.50, (
            f"Mean retrieval recall {mean_recall:.0%} is below 50% threshold. "
            f"Consider improving chunk strategy or golden QA keywords."
        )

    def test_no_question_has_zero_recall(self, recall_scores):
        """No question should have zero retrieval recall (total miss)."""
        zeros = [s for s in recall_scores if s["recall"] == 0]
        assert len(zeros) == 0, (
            f"{len(zeros)} question(s) had zero recall: "
            f"{[z['id'] for z in zeros]}"
        )
