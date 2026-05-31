# 01 — HyDE (Hypothetical Document Embeddings)

## Overview

**Problem solved**: Short, vague user queries embed very differently from the documents
that contain the answer. "Python creator" doesn't embed near "Python was created by Guido
van Rossum in 1991" — there's a semantic gap.

**HyDE solution**: Before searching, ask the LLM to generate a *hypothetical answer*.
The LLM's hallucinated answer uses the same vocabulary as the real document, so it embeds
close to the real answer and retrieves it successfully.

```
Naive: "Python creator" ──────────────────────────────────── poor match
HyDE:  "Python was created by Guido van Rossum in 1991" ──── great match
           ↑ LLM-generated, not real — only used for embedding
```

## Learning Objectives

- Understand the embedding gap between short queries and rich documents
- Implement HyDE: generate → embed → search
- Compare naive vs HyDE retrieval on the same queries
- Identify when HyDE helps (short queries, technical jargon) and when it doesn't (already detailed queries)

## How to Run

```bash
python hyde_rag.py
pytest tests/ -v
```

## Key Concept

HyDE adds one extra LLM call before retrieval. This is a cost/quality tradeoff.
For production systems, use HyDE selectively — only when query quality is reliably poor.

## Exercises

1. Test with a very specific query vs a vague one — observe the difference in improvement.
2. Generate 3 different hypothetical answers and average their embeddings (HyDE variant).
3. Measure latency overhead of the extra LLM call.
