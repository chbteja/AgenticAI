# 01 — Tokens and Embeddings

## Overview

LLMs don't read words — they read **tokens**. Understanding tokenization explains:
- Why certain inputs cost more than others
- Why word boundaries behave unexpectedly
- How context windows limit conversations

Embeddings convert tokens into **vectors in high-dimensional space** where semantic
similarity corresponds to geometric proximity. This is the foundation of every RAG system.

## Learning Objectives

- Understand what a token is and how text is split by a tokenizer
- Count tokens before making API calls (cost estimation)
- Generate vector embeddings using the OpenAI API
- Compute cosine similarity between embedding vectors
- Build a minimal semantic search function

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `tiktoken` | ≥0.7 | OpenAI's tokenizer — fast, accurate |
| `anthropic` | ≥0.40 | Anthropic token counting endpoint |
| `openai` | ≥1.50 | Embedding generation |
| `python-dotenv` | ≥1.0 | Load API keys from `.env` |

## Setup

```bash
cd phase1_foundation/01_tokens_and_embeddings
pip install -r ../../requirements.txt
cp ../../.env.example ../../.env   # then fill in your keys
```

## How to Run

```bash
# Token demo (no API key needed for tiktoken)
python tokens_demo.py

# Embedding demo (requires OPENAI_API_KEY)
python embeddings_demo.py

# Tests (all mocked — no API key needed)
pytest tests/ -v
```

## Key Concepts

- **Token**: the atomic unit an LLM processes. Roughly 1 token ≈ 0.75 words in English.
- **Tokenizer**: algorithm mapping text ↔ integer IDs. Different models use different tokenizers.
- **Context window**: maximum tokens a model can process in one call.
- **Embedding**: a fixed-length float vector representing the semantic meaning of text.
- **Cosine similarity**: angle-based distance metric; 1.0 = identical direction, 0.0 = orthogonal.

## Expected Output

```
=== Token Demo ===
"Hello, world!"              →   4 tokens
"Tokenization is complex."   →   5 tokens
"日本語テキスト"              →  11 tokens
"1 + 1 = 2"                  →   7 tokens

=== Semantic Search Demo ===
Query: "What is the capital of France?"
  1. Paris is the capital of France.           similarity=0.94
  2. The Eiffel Tower is in Paris.             similarity=0.81
  3. Python was created in 1991.               similarity=0.23
```

## Exercises

1. Add 5 sentences on a topic of your choice and observe how similar sentences cluster.
2. Modify `estimate_cost()` to include output tokens in the calculation.
3. Try tokenizing code snippets — observe that keywords often get single tokens.
