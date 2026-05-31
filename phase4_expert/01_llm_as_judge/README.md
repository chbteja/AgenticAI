# 01 — LLM-as-a-Judge

## Overview

LLM-as-a-judge uses a separate, capable LLM to score the outputs of your system on
multiple quality dimensions. It's the most scalable approach to AI evaluation —
faster and cheaper than human evaluation, and more nuanced than keyword matching.

## Evaluation Dimensions

| Metric | Question | Score |
|--------|----------|-------|
| **Faithfulness** | Is the answer grounded in the provided context? | 1–5 |
| **Relevance** | Does the answer actually address the question? | 1–5 |
| **Completeness** | Is the answer missing important information from the context? | 1–5 |
| **Coherence** | Is the answer well-structured and readable? | 1–5 |

## Learning Objectives

- Build a multi-dimension LLM evaluator
- Parse structured scores from LLM responses reliably
- Run batch evaluation over a test set
- Aggregate scores and identify the weakest dimension

## How to Run

```bash
python llm_judge.py
pytest tests/ -v
```

## Key Concept

The judge LLM should be different from (or at least prompted differently than) the
generator LLM to avoid self-serving bias. In production, use a strong model (Opus/GPT-4)
as judge even if you generate with a cheaper model.

## Exercises

1. Add a "Toxicity" dimension.
2. Compute inter-rater reliability by running the same eval twice and comparing scores.
3. Compare scores for naive RAG vs HyDE on the same test set.
