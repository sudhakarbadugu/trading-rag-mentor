#!/usr/bin/env python3
"""
RAG Pipeline Evaluation Script
===============================
Evaluates the trading-rag-mentor pipeline on three dimensions:

1. **Retrieval Recall**  — Do the retrieved chunks contain the expected keywords?
2. **Answer Faithfulness** — Is the generated answer grounded only in the context? (LLM-as-judge)
3. **Answer Relevance**   — Does the answer actually address the question?        (LLM-as-judge)

Optional: pass --ragas to also run the Ragas standardised metrics suite
(context_precision, context_recall, faithfulness, response_relevancy,
factual_correctness) after the custom eval and get a combined comparison table.

Usage:
    python scripts/evaluate_rag.py                       # custom eval (requires GROQ_API_KEY)
    python scripts/evaluate_rag.py --retrieval-only      # retrieval recall only (no LLM needed)
    python scripts/evaluate_rag.py --ragas               # custom eval + Ragas metrics
    python scripts/evaluate_rag.py --ragas --no-reference  # Ragas without reference metrics

Results are printed as a table and saved to scripts/evaluation_results.json.
Ragas results (when --ragas is used) are saved to scripts/ragas_results.json.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ── Resolve paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
GOLDEN_QA_PATH = ROOT_DIR / "tests" / "golden_qa.json"
RESULTS_PATH = ROOT_DIR / "scripts" / "evaluation_results.json"
DB_DIR = ROOT_DIR / "chroma_db"
K = 6  # chunks to retrieve per query

FAITHFULNESS_PROMPT = PromptTemplate.from_template("""You are an impartial evaluator. Given a CONTEXT and an ANSWER, determine whether every claim in the ANSWER is supported by the CONTEXT.

Rules:
- Score 1.0 if ALL claims are directly supported by the context.
- Score 0.5 if MOST claims are supported but some minor claims lack evidence.
- Score 0.0 if the answer contains claims clearly NOT in the context (hallucination).

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<one-sentence explanation>"}}""")

RELEVANCE_PROMPT = PromptTemplate.from_template("""You are an impartial evaluator. Given a QUESTION and an ANSWER, determine whether the answer is relevant and addresses the question.

Rules:
- Score 1.0 if the answer directly and completely addresses the question.
- Score 0.5 if the answer partially addresses the question or is tangential.
- Score 0.0 if the answer does not address the question at all.

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<one-sentence explanation>"}}""")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_golden_qa() -> list[dict]:
    """
    Load the human-curated golden QA dataset.

    Returns:
        list[dict]: List of evaluative question/answer/keyword entries.
    """
    with open(GOLDEN_QA_PATH) as f:
        return json.load(f)


def load_vectorstore() -> Chroma:
    """
    Load the ChromaDB vector store for evaluation.

    Returns:
        Chroma: The persistent vectorstore instance.
    """
    if not DB_DIR.exists() or not any(DB_DIR.iterdir()):
        logger.error("chroma_db/ not found. Run the app once to build the index.")
        sys.exit(1)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)


def compute_retrieval_recall(retrieved_text: str, expected_keywords: list[str]) -> float:
    """
    Calculate the fraction of expected keywords present in the retrieved content.

    Args:
        retrieved_text (str): Concatenated text from all retrieved chunks.
        expected_keywords (list[str]): List of keywords that should be present.

    Returns:
        float: Recall score between 0.0 and 1.0.
    """
    retrieved_lower = retrieved_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in retrieved_lower)
    return round(hits / len(expected_keywords), 3) if expected_keywords else 0.0


def parse_judge_response(raw: str) -> dict:
    """
    Parse the JSON output from the LLM judge, cleaning markdown if necessary.

    Args:
        raw (str): The raw string response from the LLM.

    Returns:
        dict: Parsed JSON with 'score' and 'reason'.
    """
    cleaned = raw.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (the fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse judge response: {cleaned[:200]}")
        return {"score": 0.0, "reason": "Failed to parse judge response"}


def run_llm_judge(llm, prompt_template: PromptTemplate, **kwargs) -> dict:
    """
    Invoke the LLM as an impartial judge for a given metric.

    Args:
        llm: The LangChain LLM instance.
        prompt_template (PromptTemplate): The evaluation prompt.
        **kwargs: Variables to format into the template.

    Returns:
        dict: Parsed score and reasoning.
    """
    formatted = prompt_template.format(**kwargs)
    response = llm.invoke(formatted)
    return parse_judge_response(response.content)


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate(
    retrieval_only: bool = False,
    run_ragas: bool = False,
    ragas_no_reference: bool = False,
):
    """
    Execute the RAG evaluation pipeline.

    Args:
        retrieval_only (bool): If True, skip LLM-based metrics (faithfulness/relevance).
        run_ragas (bool): If True, also run the Ragas standardised metrics suite
            after the custom evaluation and print a combined comparison table.
        ragas_no_reference (bool): When run_ragas=True, skip metrics that require a
            reference answer (context_precision, context_recall, factual_correctness).

    Returns:
        dict: Summary statistics of the custom evaluation run.  When run_ragas=True
              a ``ragas_summary`` key is also added with the Ragas aggregate scores.
    """
    golden_qa = load_golden_qa()
    vectorstore = load_vectorstore()

    # Optionally load LLM for faithfulness + relevance scoring
    llm = None
    if not retrieval_only:
        from langchain_groq import ChatGroq
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY not set. Use --retrieval-only or set it in .env")
            sys.exit(1)
        llm = ChatGroq(
            temperature=0.0,
            model_name=os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
            groq_api_key=groq_api_key,
        )

    # Load the trading mentor prompt for generation
    from prompts import TRADING_MENTOR_PROMPT
    rag_prompt = PromptTemplate.from_template(TRADING_MENTOR_PROMPT)

    results = []
    logger.info(f"\n{'='*80}")
    logger.info(f"  RAG Evaluation — {len(golden_qa)} questions")
    logger.info(f"  Mode: {'Retrieval Only' if retrieval_only else 'Full (Retrieval + LLM Judge)'}")
    logger.info(f"{'='*80}\n")

    for i, qa in enumerate(golden_qa, 1):
        qid = qa["id"]
        question = qa["question"]
        logger.info(f"[{i}/{len(golden_qa)}] {qid}: {question}")

        # ── Step 1: Retrieve ──────────────────────────────────────────────
        scored_docs = vectorstore.similarity_search_with_relevance_scores(question, k=K)
        retrieved_text = " ".join(doc.page_content.lower() for doc, _ in scored_docs)
        top_score = max((s for _, s in scored_docs), default=0.0)

        retrieval_recall = compute_retrieval_recall(
            retrieved_text, qa["expected_context_keywords"]
        )
        logger.info(f"  Retrieval recall: {retrieval_recall:.0%}  |  Top similarity: {top_score:.3f}")

        entry = {
            "id": qid,
            "question": question,
            "category": qa["category"],
            "retrieval_recall": retrieval_recall,
            "top_similarity_score": round(top_score, 3),
        }

        # ── Step 2: Generate + Judge (if not retrieval-only) ──────────────
        if not retrieval_only and llm is not None:
            context = "\n\n".join(doc.page_content for doc, _ in scored_docs)
            formatted_prompt = rag_prompt.format(context=context, question=question, chat_history="")
            answer = llm.invoke(formatted_prompt).content

            # Faithfulness check
            faith_result = run_llm_judge(llm, FAITHFULNESS_PROMPT, context=context, answer=answer)
            entry["faithfulness"] = faith_result["score"]
            entry["faithfulness_reason"] = faith_result["reason"]
            logger.info(f"  Faithfulness:    {faith_result['score']:.1f}  — {faith_result['reason']}")

            # Relevance check
            rel_result = run_llm_judge(llm, RELEVANCE_PROMPT, question=question, answer=answer)
            entry["relevance"] = rel_result["score"]
            entry["relevance_reason"] = rel_result["reason"]
            logger.info(f"  Relevance:       {rel_result['score']:.1f}  — {rel_result['reason']}")

            entry["generated_answer_preview"] = answer[:300]

        results.append(entry)
        logger.info("")

    # ── Aggregate scores ──────────────────────────────────────────────────
    n = len(results)
    avg_retrieval = sum(r["retrieval_recall"] for r in results) / n
    logger.info(f"{'='*80}")
    logger.info(f"  AGGREGATE SCORES ({n} questions)")
    logger.info(f"{'='*80}")
    logger.info(f"  Mean Retrieval Recall:  {avg_retrieval:.1%}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": n,
        "k": K,
        "mean_retrieval_recall": round(avg_retrieval, 3),
    }

    if not retrieval_only:
        scores_with_faith = [r for r in results if "faithfulness" in r]
        scores_with_rel = [r for r in results if "relevance" in r]
        if scores_with_faith:
            avg_faith = sum(r["faithfulness"] for r in scores_with_faith) / len(scores_with_faith)
            summary["mean_faithfulness"] = round(avg_faith, 3)
            logger.info(f"  Mean Faithfulness:      {avg_faith:.1%}")
        if scores_with_rel:
            avg_rel = sum(r["relevance"] for r in scores_with_rel) / len(scores_with_rel)
            summary["mean_relevance"] = round(avg_rel, 3)
            logger.info(f"  Mean Relevance:         {avg_rel:.1%}")

    logger.info("")

    # ── Save results ──────────────────────────────────────────────────────
    output = {"summary": summary, "results": results}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Results saved to {RESULTS_PATH}")

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n{'ID':<30} {'Category':<18} {'Retrieval':<12}", end="")
    if not retrieval_only:
        print(f"{'Faith.':<10} {'Relev.':<10}", end="")
    print()
    print("-" * (80 if not retrieval_only else 60))

    for r in results:
        print(f"{r['id']:<30} {r['category']:<18} {r['retrieval_recall']:<12.0%}", end="")
        if not retrieval_only:
            faith = r.get("faithfulness", "-")
            rel = r.get("relevance", "-")
            faith_str = f"{faith:.1f}" if isinstance(faith, float) else faith
            rel_str = f"{rel:.1f}" if isinstance(rel, float) else rel
            print(f"{faith_str:<10} {rel_str:<10}", end="")
        print()

    print("-" * (80 if not retrieval_only else 60))
    print(f"{'MEAN':<30} {'':<18} {avg_retrieval:<12.0%}", end="")
    if not retrieval_only:
        print(f"{summary.get('mean_faithfulness', '-'):<10} {summary.get('mean_relevance', '-'):<10}", end="")
    print("\n")

    # ── Optional Ragas evaluation ─────────────────────────────────────────
    if run_ragas:
        if retrieval_only:
            logger.warning("--ragas is ignored when --retrieval-only is set (no answers generated).")
        else:
            logger.info("\nStarting Ragas evaluation…\n")
            try:
                from ragas_eval import run_ragas_evaluation  # same scripts/ dir
                ragas_summary = run_ragas_evaluation(
                    skip_reference_metrics=ragas_no_reference,
                )
                summary["ragas_summary"] = ragas_summary

                # ── Combined comparison table ─────────────────────────────
                print(f"\n{'='*100}")
                print("  COMBINED COMPARISON  (Custom LLM-judge  vs  Ragas standardised metrics)")
                print(f"{'='*100}")
                print(
                    f"{'ID':<30} {'Category':<18} "
                    f"{'[Custom]':^30}  "
                    f"{'[Ragas]':^40}"
                )
                print(
                    f"{'':30} {'':18} "
                    f"{'Recall':<12}{'Faith.':<10}{'Relev.':<10}  "
                    f"{'Faith.':<10}{'Resp.Rel.':<12}{'CTX_P':<10}{'CTX_R':<10}{'Factual':<10}"
                )
                print("-" * 100)

                # Load Ragas per-question scores
                ragas_results_path = ROOT_DIR / "scripts" / "ragas_results.json"
                ragas_per_q: dict[str, dict] = {}
                if ragas_results_path.exists():
                    with open(ragas_results_path) as f:
                        ragas_data = json.load(f)
                    for rec in ragas_data.get("per_question", []):
                        ragas_per_q[rec["id"]] = rec

                # Column names that Ragas may use (varies slightly by version)
                def _rget(rec: dict, *keys, default: float = float("nan")) -> float:
                    for k in keys:
                        if k in rec:
                            return float(rec[k])
                    return default

                for r in results:
                    rq = ragas_per_q.get(r["id"], {})
                    row = (
                        f"{r['id']:<30} {r['category']:<18} "
                        f"{r['retrieval_recall']:<12.0%}"
                    )
                    # Custom judge columns
                    faith_c = r.get("faithfulness", "-")
                    rel_c   = r.get("relevance", "-")
                    row += f"{(f'{faith_c:.1f}' if isinstance(faith_c, float) else '-'):<10}"
                    row += f"{(f'{rel_c:.1f}'   if isinstance(rel_c,   float) else '-'):<10}  "
                    # Ragas columns
                    row += f"{_rget(rq, 'faithfulness'):<10.3f}"
                    row += f"{_rget(rq, 'answer_relevancy'):<12.3f}"
                    row += f"{_rget(rq, 'context_precision_with_reference', 'context_precision'):<10.3f}"
                    row += f"{_rget(rq, 'context_recall'):<10.3f}"
                    row += f"{_rget(rq, 'factual_correctness(mode=f1)', 'factual_correctness', 'answer_correctness'):<10.3f}"
                    print(row)

                print("-" * 100)
                # Means row
                rs = ragas_summary
                means = (
                    f"{'MEAN':<30} {'':18} "
                    f"{avg_retrieval:<12.0%}"
                    f"{summary.get('mean_faithfulness', '-'):<10} "
                    f"{summary.get('mean_relevance', '-'):<10}  "
                    f"{rs.get('mean_faithfulness', float('nan')):<10.3f}"
                    f"{rs.get('mean_answer_relevancy', float('nan')):<12.3f}"
                    f"{rs.get('mean_context_precision', float('nan')):<10.3f}"
                    f"{rs.get('mean_context_recall', float('nan')):<10.3f}"
                    f"{rs.get('mean_factual_correctness', float('nan')):<10.3f}"
                )
                print(means + "\n")

            except ImportError as exc:
                logger.error(
                    f"Ragas is not installed: {exc}\n"
                    "Install it with:  pip install 'ragas>=0.2' datasets"
                )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval recall (no LLM calls needed)",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Also run Ragas standardised metrics after the custom eval.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="When --ragas is set, skip reference-dependent Ragas metrics "
             "(context_precision, context_recall, factual_correctness).",
    )
    args = parser.parse_args()
    evaluate(
        retrieval_only=args.retrieval_only,
        run_ragas=args.ragas,
        ragas_no_reference=args.no_reference,
    )
