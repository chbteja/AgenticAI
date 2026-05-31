# 02 — Naive RAG Pipeline

## Overview

RAG (Retrieval-Augmented Generation) solves a fundamental LLM limitation: models can
only answer questions about what they saw during training. RAG lets you inject fresh,
private, or specialised knowledge at query time by:
1. Pre-processing documents into a searchable vector store
2. At query time: retrieving the most relevant chunks
3. Sending retrieved chunks + query to the LLM as context

This is the "naive" baseline. It works, but has known failure modes (see Exercises).
Phase 3 fixes each one.

## Learning Objectives

- Load and split documents with RecursiveCharacterTextSplitter
- Generate embeddings and store in a Chroma vector database
- Retrieve relevant chunks using similarity search
- Build a complete Q&A chain: query → retrieve → generate
- Identify failure modes: chunk size, query quality, retrieval precision

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.3 | Document loaders, text splitters, chains |
| `langchain-openai` | ≥0.2 | OpenAI embeddings |
| `langchain-anthropic` | ≥0.3 | Generation LLM |
| `langchain-chroma` | ≥0.1 | Vector store integration |
| `chromadb` | ≥0.5 | Vector database |
| `pypdf` | ≥4.0 | PDF loading |

## Setup

```bash
# Requires both OPENAI_API_KEY (for embeddings) and ANTHROPIC_API_KEY (for generation)
```

## How to Run

```bash
# Build the vector store and run Q&A
python naive_rag.py

# Run with a custom question
python naive_rag.py --question "What is the main topic of the document?"

# Tests (fully mocked)
pytest tests/ -v
```

## RAG Pipeline Architecture

```
documents/
    sample.txt
        │
        ▼
DocumentLoader → raw Documents
        │
        ▼ RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
Chunks [doc1, doc2, doc3, ...]
        │
        ▼ OpenAIEmbeddings
Vectors [(0.1, 0.3, ...), ...]
        │
        ▼ Chroma.from_documents()
Vector Store (persisted to ./chroma_db)
        │
   Query: "..."
        │
        ▼ similarity_search(query, k=4)
Top-K Chunks
        │
        ▼ ChatAnthropic
Answer
```

## Key Parameters to Understand

| Parameter | Default | Effect if too small | Effect if too large |
|-----------|---------|--------------------|--------------------|
| `chunk_size` | 500 | Loses sentence context | Dilutes relevant signal |
| `chunk_overlap` | 50 | Cuts sentences at boundaries | Higher storage cost |
| `k` (top-k retrieval) | 4 | Misses relevant context | Sends too much noise to LLM |

## Expected Output

```
Building vector store from 5 document chunks...
Vector store ready.

Q: What programming language is described?
A: Python is described as a versatile programming language created by Guido van Rossum.
```

## Exercises

1. Set `chunk_size=100` and `chunk_size=2000` — compare answer quality.
2. Ask a question whose answer spans two chunks — observe that it fails.
3. Count how many chunks are retrieved and whether all are relevant.
