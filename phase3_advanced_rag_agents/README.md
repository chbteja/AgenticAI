# Phase 3 — Advanced RAG + Agents (Weeks 6–9)

## Overview

Naive RAG fails in predictable ways. This phase teaches the techniques that fix each failure,
then introduces AI agents — systems that can use tools, reason across multiple steps, and
handle tasks too complex for a single LLM call.

## Failure → Fix Mapping

| Naive RAG Failure | Advanced Technique | Module |
|-------------------|--------------------|--------|
| Poor/vague queries don't match relevant chunks | HyDE (Hypothetical Document Embeddings) | `01_hyde` |
| Single query misses multiple relevant perspectives | Multi-query retrieval | `02_multi_query` |
| Small chunks lose surrounding context | Parent-document retriever | `03_parent_doc_retriever` |
| Retrieved chunks ranked suboptimally | Cross-encoder reranking | `04_reranking` |

## Agents

| Pattern | Framework | Module |
|---------|-----------|--------|
| ReAct (Reason + Act) loop | LangChain tools | `05_react_agents` |
| Production-grade agent with state | LangGraph | `06_langgraph` |

## Learning Path

1. Run each RAG technique on the same question and compare outputs
2. Build a ReAct agent with 2-3 custom tools
3. Migrate the ReAct agent to LangGraph — observe how explicit state management changes the code
4. Add a conditional edge to your LangGraph agent (e.g. "if answer is unclear, search again")

## Key Insight

LangGraph is the right way to build ReAct agents in production. The older `AgentExecutor`
class is harder to debug and customise. With LangGraph, the control flow is explicit code,
not a black box.
