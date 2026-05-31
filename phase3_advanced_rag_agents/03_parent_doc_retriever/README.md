# 03 — Parent Document Retriever

## Overview

**Problem solved**: Small chunks are precise for retrieval but lose surrounding context.
Large chunks give better context but embed poorly (the embedding averages too much signal).

**Solution**: Index small chunks for precise retrieval, but when a small chunk matches,
return its larger *parent* chunk to the LLM for context. Best of both worlds.

```
Indexing:  [parent_doc_1] → split → [small_1a] [small_1b] [small_1c]
                                       ↓ embed      ↓ embed      ↓ embed
                                      stored in   stored in   stored in
                                      vector DB   vector DB   vector DB
                                       ↓ pointer back to parent ↓
Retrieval: query matches small_1b → return parent_doc_1 to LLM
```

## Learning Objectives

- Understand the chunk size trade-off for retrieval vs context quality
- Set up ParentDocumentRetriever with an InMemoryStore
- Observe that the LLM receives parent-sized context even when small chunks match

## How to Run

```bash
python parent_doc_rag.py
pytest tests/ -v
```

## Key Concept

The `ParentDocumentRetriever` uses two splitters:
- `child_splitter`: small chunks (100–300 chars) indexed in the vector store
- `parent_splitter` (optional): medium chunks (1000–2000 chars) returned to the LLM

Without `parent_splitter`, the entire original document is the parent.
