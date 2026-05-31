# Phase 2 — Core LangChain + Naive RAG (Weeks 3–6)

## Overview

LangChain is a framework for composing LLM applications. Its modern interface — LCEL
(LangChain Expression Language) — uses the `|` pipe operator to chain components together
just like Unix pipes. This phase teaches LCEL properly, then uses it to build your first
complete RAG (Retrieval-Augmented Generation) pipeline.

## Modules

| Module | What You Build | Key Skill |
|--------|---------------|-----------|
| `01_lcel_basics` | Chains, parallel execution, passthrough | LCEL composition patterns |
| `02_naive_rag` | Full RAG pipeline: PDF → chunks → embed → store → retrieve → answer | End-to-end RAG |

## Learning Path

1. **LCEL first**: master `prompt | llm | parser` chains before adding retrieval
2. **Naive RAG**: build and run the pipeline, then intentionally break it (bad queries, wrong chunk size) to understand its failure modes
3. Before moving to Phase 3, write down 3 ways naive RAG fails on your test documents — Phase 3 fixes each one

## Key Insight

Naive RAG fails in predictable ways:
- Short, vague queries don't match the right chunks (fix: HyDE, multi-query)
- Small chunks lose context (fix: parent-doc retriever)
- Retrieved chunks may be relevant but ranked poorly (fix: reranking)

Understanding the failure modes is more important than the pipeline itself.
