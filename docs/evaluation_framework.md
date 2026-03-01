# RAG Evaluation Framework

## Why evaluate a RAG pipeline?

Building a RAG system is step one. Proving it works _reliably_ is step two — and the step most portfolios skip. Evaluation answers three fundamental questions:

1. **Is the retriever finding the right context?** (Retrieval quality)
2. **Is the LLM staying faithful to that context?** (Hallucination prevention)
3. **Is the answer actually useful to the user?** (Relevance)

---

## The Three Metrics

### 1. Retrieval Recall

> _"Of the keywords we expect to see in the context, how many actually appear?"_

**How it works:**

- Each golden QA entry has `expected_context_keywords` — domain terms that should appear in the retrieved chunks.
- After retrieval, we check what fraction of those keywords appear in the combined retrieved text.
- Score: `matched_keywords / total_expected_keywords` (0.0 to 1.0)

**Why this metric:**

- It directly measures whether ChromaDB + HuggingFace embeddings are surfacing the right transcript segments.
- Keyword matching is simple but effective for domain-specific content where terminology is precise.

**Limitations:**

- Synonyms or paraphrases won't be caught (e.g., "stop loss" vs "risk limit").
- A high recall doesn't guarantee the chunks are sufficient for a complete answer.

---

### 2. Answer Faithfulness (LLM-as-Judge)

> _"Is every claim in the answer supported by the retrieved context?"_

**How it works:**

- After generating an answer, we ask a separate LLM call (the "judge") to evaluate whether each claim traces back to the context.
- The judge scores: 1.0 (fully faithful), 0.5 (mostly faithful), 0.0 (hallucinating).

**Why this metric:**

- Faithfulness directly measures hallucination — the #1 failure mode of RAG systems.
- Our prompt already says "Use ONLY the information present in the provided excerpts," but we need to _verify_ the LLM actually obeys.

**The judge prompt:**

```
Given a CONTEXT and an ANSWER, determine whether every claim in the
ANSWER is supported by the CONTEXT.
```

---

### 3. Answer Relevance (LLM-as-Judge)

> _"Does the answer actually address the question?"_

**How it works:**

- A separate LLM call evaluates whether the answer is on-topic and addresses the user's question.
- Score: 1.0 (directly answers), 0.5 (partially answers), 0.0 (off-topic).

**Why this metric:**

- A faithful answer can still be irrelevant if the retriever surfaces wrong chunks.
- This catches cases where the answer is "grounded" but the grounding is in the wrong part of the corpus.

---

## Golden QA Dataset

The golden QA dataset (`tests/golden_qa.json`) contains 12 question/keyword triplets spanning five categories:

| Category          | Questions | Topics                                                 |
| ----------------- | --------- | ------------------------------------------------------ |
| `vcp`             | 2         | VCP definition, line of least resistance               |
| `entries`         | 3         | 200-day MA rule, breakout volume, primary base         |
| `risk_management` | 2         | Stop-loss rules, position sizing                       |
| `fundamentals`    | 3         | Market leaders, relative strength, cyclicals, earnings |
| `mindset`         | 1         | Style drift                                            |

### Design principles:

- Every question is answerable from the actual transcript content (Days 1-5).
- Keywords are specific domain terms, not generic words.
- Categories cover the breadth of the trading methodology.

---

## How to Run

### Retrieval-only (fast, no API key needed)

```bash
python scripts/evaluate_rag.py --retrieval-only
```

### Full evaluation (requires GROQ_API_KEY)

```bash
python scripts/evaluate_rag.py
```

### Pytest (dataset validation + retrieval recall)

```bash
python -m pytest tests/test_rag_evaluation.py -v
```

---

## Extending the Evaluation

### Adding new questions

1. Add an entry to `tests/golden_qa.json` with all required fields.
2. Re-run the evaluation to check the new entry.

### Swapping the judge model

Change the `model_name` in `scripts/evaluate_rag.py` — any Groq-hosted model works. Lower-temperature models (0.0) give more consistent judge scores.

### Industry frameworks

For more rigorous evaluation at scale, consider:

- **[RAGAS](https://docs.ragas.io/)** — automated RAG evaluation with faithfulness, relevance, and context metrics
- **[DeepEval](https://docs.deepeval.com/)** — unit testing for LLM outputs with built-in metrics

These frameworks provide more granular scoring (claim-level faithfulness, context precision/recall) but require more setup. Our lightweight approach achieves the same conceptual coverage for a portfolio project.

---

_Last updated: 2026-03-01_
