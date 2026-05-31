# 01 — LCEL (LangChain Expression Language) Basics

## Overview

LCEL is the modern, recommended way to compose LangChain components. The `|` pipe
operator chains `Runnable` objects together. Each runnable's output becomes the next
one's input. This gives you streaming, batching, and async for free.

## Learning Objectives

- Understand the `Runnable` interface and why everything in LangChain implements it
- Build simple `prompt | llm | parser` chains
- Use `RunnableParallel` to run multiple chains concurrently
- Use `RunnablePassthrough` to pass data through unchanged
- Stream LCEL chains token-by-token

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.3 | Core framework |
| `langchain-anthropic` | ≥0.3 | Anthropic model integration |
| `langchain-core` | ≥0.3 | PromptTemplate, OutputParsers |
| `python-dotenv` | ≥1.0 | API key loading |

## How to Run

```bash
python lcel_chains.py
pytest tests/ -v
```

## Key Concepts

- `prompt | llm` — builds a chain; `|` calls `__or__` which returns a `RunnableSequence`
- `chain.invoke({"var": "value"})` — run synchronously
- `chain.stream(...)` — iterate over token chunks
- `chain.batch([...])` — run multiple inputs concurrently
- `RunnableParallel(a=chain_a, b=chain_b)` — run both chains, return `{"a": ..., "b": ...}`
- `RunnablePassthrough()` — identity transform; useful for injecting context

## Expected Output

```
=== Simple Chain ===
Paris is the capital of France.

=== Parallel Chain ===
summary:     Python is a general-purpose language...
translation: Python est un langage polyvalent...

=== Streaming ===
The speed of light is 299,792,458 metres per second...
```

## Exercises

1. Add a `| StrOutputParser()` to strip metadata from responses.
2. Build a chain that first generates a joke, then explains it.
3. Use `.batch()` to process 5 questions simultaneously and measure time vs sequential.
