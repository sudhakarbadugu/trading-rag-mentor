#!/usr/bin/env python3
"""
Ragas RAG Evaluation
====================
Evaluates the trading-rag-mentor pipeline using the Ragas library (>= 0.2),
which provides standardised, LLM-backed metrics that complement the custom
LLM-as-judge in evaluate_rag.py.

Ragas metrics evaluated
-----------------------
Metric                  Needs reference?  What it measures
----------------------  ----------------  ---------------------------------------------------
faithfulness            No                Are all response claims grounded in the context?
response_relevancy      No                Does the response directly address the question?
context_precision       Yes               Are retrieved chunks ranked by usefulness?
context_recall          Yes               Does the context cover all reference answer content?
factual_correctness     Yes               Is the response factually consistent with reference?

Prerequisites
-------------
    GROQ_API_KEY set in .env (or environment)
    pip install ragas>=0.2 datasets

Usage
-----
    # Full evaluation on all golden QA questions
    python scripts/ragas_eval.py

    # Smoke-test on first N questions only
    python scripts/ragas_eval.py --sample 3

    # Skip reference-dependent metrics (faster, no ground-truth required)
    python scripts/ragas_eval.py --no-reference

Results are saved to scripts/ragas_results.json and also printed as a table.

Importing from evaluate_rag.py
------------------------------
    from ragas_eval import run_ragas_evaluation
    summary = run_ragas_evaluation(sample_n=None, skip_reference_metrics=False)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

GOLDEN_QA_PATH = ROOT_DIR / "tests" / "golden_qa.json"
RESULTS_PATH   = ROOT_DIR / "scripts" / "ragas_results.json"
DB_DIR         = ROOT_DIR / "data" / "chroma_db"
K = 6  # chunks to retrieve per query

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_golden_qa() -> list[dict]:
    with open(GOLDEN_QA_PATH) as f:
        return json.load(f)


def _load_vectorstore():
    """Return a ChromaDB vectorstore, exiting on error."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    if not DB_DIR.exists() or not any(DB_DIR.iterdir()):
        logger.error("data/chroma_db not found — run the app once to build the index.")
        sys.exit(1)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)


def _build_llm():
    """Instantiate the Groq LLM, exiting if the API key is absent."""
    from langchain_groq import ChatGroq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        logger.error("GROQ_API_KEY is not set. Add it to .env or export it.")
        sys.exit(1)
    return ChatGroq(
        temperature=0.0,
        model_name=os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
        groq_api_key=key,
    )


def _check_ragas_importable() -> None:
    """Raise an ImportError with an install hint if ragas is missing."""
    try:
        import ragas  # noqa: F401
        from ragas.metrics.collections import Faithfulness  # noqa: F401
    except ImportError:
        raise ImportError(
            "ragas >= 0.2 is required.  "
            "Install it with:  pip install 'ragas>=0.2' datasets"
        )


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_ragas_evaluation(
    sample_n: int | None = None,
    skip_reference_metrics: bool = False,
) -> dict:
    """
    Run Ragas evaluation over the golden QA dataset.

    Args:
        sample_n: When set, evaluate only the first *n* questions (useful for
                  smoke-testing without spending LLM tokens on the full set).
        skip_reference_metrics: When True, skip context_precision, context_recall
                                 and factual_correctness (faster; no ground-truth
                                 reference needed).

    Returns:
        dict: Summary statistics keyed by metric name, plus ``timestamp`` /
              ``num_questions`` / ``k`` metadata.
    """
    _check_ragas_importable()

    from ragas.dataset_schema import SingleTurnSample  # noqa: F401 (kept for typing)
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        FactualCorrectness,
    )
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
    from langchain_core.prompts import PromptTemplate
    from prompts import TRADING_MENTOR_PROMPT

    # ── Initialise models ─────────────────────────────────────────────────
    logger.info("Initialising models…")

    # LangChain LLM — used only for RAG answer generation
    gen_llm = _build_llm()

    # Ragas LLM — llm_factory with Groq served on the OpenAI-compatible endpoint
    groq_api_key = os.environ.get("GROQ_API_KEY")
    model_name = os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    try:
        from openai import AsyncOpenAI as _AsyncOpenAIClient
    except ImportError:
        raise ImportError("openai package is required for Ragas. pip install openai")
    # Ragas collections metrics use acomplete() internally → requires an async client
    groq_compat_client = _AsyncOpenAIClient(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    ragas_llm = llm_factory(model_name, provider="openai", client=groq_compat_client, max_tokens=16384)

    # Ragas embeddings — ragas' native HuggingFace wrapper (not LangChain's)
    ragas_emb = RagasHFEmbeddings(model="all-MiniLM-L6-v2")

    # ── Select metrics ────────────────────────────────────────────────────
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ]
    if not skip_reference_metrics:
        metrics += [
            ContextPrecisionWithReference(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
            FactualCorrectness(llm=ragas_llm),
        ]

    # ── Load data ─────────────────────────────────────────────────────────
    golden_qa = _load_golden_qa()
    if sample_n:
        golden_qa = golden_qa[:sample_n]

    entries_without_ref = [e["id"] for e in golden_qa if not e.get("reference_answer")]
    if entries_without_ref and not skip_reference_metrics:
        logger.warning(
            f"{len(entries_without_ref)} entries have no 'reference_answer'. "
            "Reference-dependent metrics may be inaccurate for those entries."
        )

    vectorstore = _load_vectorstore()
    rag_prompt  = PromptTemplate.from_template(TRADING_MENTOR_PROMPT)

    # ── Generate answers ─────────────────────────────────────────────────
    logger.info(f"\n{'='*80}")
    logger.info(f"  Ragas Evaluation — {len(golden_qa)} questions")
    logger.info(f"  Metrics: {[type(m).__name__ for m in metrics]}")
    logger.info(f"{'='*80}\n")

    rows: list[dict] = []   # {user_input, response, retrieved_contexts, reference, id, category}

    for i, qa in enumerate(golden_qa, 1):
        logger.info(f"[{i}/{len(golden_qa)}] Generating: {qa['question']}")

        scored_docs = vectorstore.similarity_search_with_relevance_scores(
            qa["question"], k=K
        )
        chunks  = [doc.page_content for doc, _ in scored_docs]
        context = "\n\n".join(chunks)

        # chat_history is empty for standalone evaluation
        formatted = rag_prompt.format(
            context=context, question=qa["question"], chat_history=""
        )
        answer = gen_llm.invoke(formatted).content

        rows.append({
            "id":                 qa["id"],
            "category":           qa["category"],
            "question":           qa["question"],
            "user_input":         qa["question"],
            "response":           answer,
            "retrieved_contexts": chunks,
            "reference":          qa.get("reference_answer", ""),
        })

    # ── Score each metric via batch_score() ───────────────────────────────
    # Each collections metric has a typed ascore(**specific_kwargs) — we build
    # the inputs list to match those kwargs exactly.
    METRIC_INPUTS: dict[str, list[str]] = {
        "Faithfulness":                  ["user_input", "response", "retrieved_contexts"],
        "AnswerRelevancy":               ["user_input", "response"],
        "ContextPrecisionWithReference": ["user_input", "reference", "retrieved_contexts"],
        "ContextRecall":                 ["user_input", "retrieved_contexts", "reference"],
        "FactualCorrectness":            ["response", "reference"],
    }

    logger.info("\nRunning Ragas metric scoring (may take several minutes)…")

    metric_scores: dict[str, list[float | None]] = {}   # metric_name → per-question scores

    for metric in metrics:
        metric_name = type(metric).__name__
        required_keys = METRIC_INPUTS.get(metric_name, [])
        inputs = [{k: r[k] for k in required_keys} for r in rows]
        logger.info(f"  Scoring {metric_name} on {len(inputs)} question(s)…")
        try:
            results_list = metric.batch_score(inputs)
            scores = []
            for res in results_list:
                val = res.result if hasattr(res, "result") else res
                scores.append(float(val) if val is not None else None)
        except Exception as e:
            logger.warning(f"  {metric_name} failed: {e}")
            scores = [None] * len(rows)
        metric_scores[metric_name] = scores

    # ── Aggregate ─────────────────────────────────────────────────────────
    METRIC_FRIENDLY: dict[str, str] = {
        "Faithfulness":                  "faithfulness",
        "AnswerRelevancy":               "answer_relevancy",
        "ContextPrecisionWithReference": "context_precision",
        "ContextRecall":                 "context_recall",
        "FactualCorrectness":            "factual_correctness",
    }

    summary: dict = {
        "timestamp":     datetime.now().isoformat(),
        "num_questions": len(golden_qa),
        "k":             K,
    }

    logger.info(f"\n{'='*80}")
    logger.info("  RAGAS AGGREGATE SCORES")
    logger.info(f"{'='*80}")

    for metric_cls_name, friendly in METRIC_FRIENDLY.items():
        if metric_cls_name in metric_scores:
            valid = [s for s in metric_scores[metric_cls_name] if s is not None]
            if valid:
                mean_val = sum(valid) / len(valid)
                summary[f"mean_{friendly}"] = round(mean_val, 3)
                logger.info(f"  mean_{friendly:<35} {mean_val:.3f}")

    # ── Per-question records ───────────────────────────────────────────────
    per_q_records = []
    for i, row in enumerate(rows):
        rec = {
            "id":       row["id"],
            "category": row["category"],
            "question": row["question"],
        }
        for metric_cls_name, friendly in METRIC_FRIENDLY.items():
            if metric_cls_name in metric_scores:
                rec[friendly] = metric_scores[metric_cls_name][i]
        per_q_records.append(rec)

    # ── Save ──────────────────────────────────────────────────────────────
    output = {"summary": summary, "per_question": per_q_records}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Results saved → {RESULTS_PATH}")

    # ── Print table ───────────────────────────────────────────────────────
    active_metrics = [
        (cls_name, friendly)
        for cls_name, friendly in METRIC_FRIENDLY.items()
        if cls_name in metric_scores
    ]
    col_header = " ".join(f"{fn[:9]:<11}" for _, fn in active_metrics)
    header = f"{'ID':<30} {col_header}"
    print(f"\n{header}")
    print("-" * len(header))

    for rec in per_q_records:
        row_str = f"{rec['id']:<30} "
        for _, fn in active_metrics:
            val = rec.get(fn)
            row_str += f"{val:<11.3f}" if isinstance(val, float) else f"{'-':<11}"
        print(row_str)

    print("-" * len(header))
    means_row = f"{'MEAN':<30} "
    for _, fn in active_metrics:
        val = summary.get(f"mean_{fn}")
        means_row += f"{val:<11.3f}" if val is not None else f"{'-':<11}"
    print(means_row + "\n")

    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline with Ragas standardised metrics."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N questions (smoke-test mode).",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip reference-dependent metrics (context_precision, context_recall, "
             "factual_correctness). Faster; no ground-truth required.",
    )
    args = parser.parse_args()
    run_ragas_evaluation(
        sample_n=args.sample,
        skip_reference_metrics=args.no_reference,
    )
