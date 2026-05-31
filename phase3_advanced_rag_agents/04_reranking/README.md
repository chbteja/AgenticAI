# 04 — Reranking with Cross-Encoders

## Overview

**Problem solved**: Vector similarity retrieves documents that are semantically related
to the query, but "related" is not the same as "answers the question". The top-3 retrieved
chunks might include one highly relevant and two tangentially related chunks — but they're
ranked by embedding similarity, not relevance to the specific question.

**Solution**: Use a cross-encoder model to *rerank* the retrieved documents. A cross-encoder
sees the query AND document together (unlike a bi-encoder which embeds them separately),
enabling much more nuanced relevance scoring.

```
Initial retrieval (bi-encoder, fast): [doc_c, doc_a, doc_b, doc_e, doc_d]
                                              ↓ cross-encoder reranking
After reranking (cross-encoder, slower): [doc_a, doc_c, doc_b, doc_d, doc_e]
                                              ↑ more relevant docs moved up
```

## Learning Objectives

- Understand bi-encoder vs cross-encoder scoring
- Use FlashrankRerank (local, free) and CohereRerank (API-based, paid)
- Implement retrieve-then-rerank pipeline
- Observe ranking changes before and after reranking

## How to Run

```bash
# FlashrankRerank (no API key needed)
python reranking_rag.py

# Tests
pytest tests/ -v
```

## Key Concept

The typical pattern is **over-retrieve then rerank**:
1. Retrieve top-20 documents (high recall)
2. Rerank to top-5 (high precision)
3. Send top-5 to LLM

This costs more than retrieving top-5 directly but produces significantly better results.
