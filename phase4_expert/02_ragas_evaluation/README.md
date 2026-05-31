# 02 — RAGAS Evaluation

## Overview

RAGAS (Retrieval Augmented Generation Assessment) is the standard framework for
evaluating RAG pipelines. It provides 4 component metrics, each targeting a different
failure mode.

## RAGAS Metrics

| Metric | Measures | Needs |
|--------|----------|-------|
| **Faithfulness** | Are answer claims supported by retrieved context? | question + answer + contexts |
| **Answer Relevancy** | Does the answer address the question? | question + answer |
| **Context Precision** | Are retrieved chunks actually useful? | question + contexts + ground truth |
| **Context Recall** | Did retrieval find all needed information? | contexts + ground truth |

## Learning Objectives

- Understand what each RAGAS metric measures
- Create a RAGAS evaluation dataset from your RAG pipeline outputs
- Run RAGAS evaluation and interpret the results
- Use scores to identify the weakest component of your RAG pipeline

## How to Run

```bash
python ragas_eval.py
pytest tests/ -v
```

## Key Concepts

- **Faithfulness < 0.8**: your generator is hallucinating beyond the retrieved context
- **Answer Relevancy < 0.8**: your generator is off-topic (prompt issue)
- **Context Precision < 0.7**: your retriever is returning irrelevant chunks
- **Context Recall < 0.7**: your retriever is missing relevant chunks (try multi-query or HyDE)

## Exercises

1. Run RAGAS on your naive RAG pipeline, then again after adding HyDE. Compare the scores.
2. Add 10 more questions to the test set — observe how scores change with sample size.
3. Write a script that flags any question with faithfulness < 0.5 for human review.
