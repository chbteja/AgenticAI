# 02 — Multi-Query Retrieval

## Overview

**Problem solved**: A single query captures only one semantic angle. If your documents
use different vocabulary than the user, or the question can be phrased multiple ways,
a single query will miss relevant content.

**Solution**: Generate N variations of the user's query (using an LLM), run all of them
through the retriever, take the union of results, deduplicate, and pass the combined
context to the final answer LLM.

```
Query: "Python web frameworks"
  ├─ Variant 1: "Python libraries for building web applications"
  ├─ Variant 2: "Flask and Django alternatives"
  └─ Variant 3: "server-side Python frameworks"
         │
         ▼ union + deduplicate
  All unique relevant docs → LLM answer
```

## Learning Objectives

- Generate semantically diverse query variants with an LLM
- Retrieve across all variants and deduplicate by document content
- Understand the recall/cost tradeoff (more queries = better recall, more API calls)

## How to Run

```bash
python multi_query_rag.py
pytest tests/ -v
```

## Key Concept

Multi-query improves **recall** — it finds more relevant documents. The tradeoff is
N extra LLM calls for query generation and N retrieval operations. In production,
use 3–5 variants; diminishing returns beyond that.
