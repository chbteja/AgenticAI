# Agentic AI Curriculum — Hands-On Examples

A progressive, project-based curriculum for mastering LLMs, RAG pipelines, and AI agents.
Each example is self-contained with runnable code, tests, and a README explaining the concepts.

---

## Curriculum Structure

```
AgenticAI/
├── phase1_foundation/          # Weeks 1–3: Tokens, Embeddings, Raw APIs, Prompt Engineering
│   ├── 01_tokens_and_embeddings/
│   ├── 02_raw_api/
│   └── 03_prompt_engineering/
├── phase2_langchain_rag/       # Weeks 3–6: LCEL, Naive RAG Pipeline
│   ├── 01_lcel_basics/
│   └── 02_naive_rag/
├── phase3_advanced_rag_agents/ # Weeks 6–9: HyDE, Multi-Query, Reranking, ReAct, LangGraph
│   ├── 01_hyde/
│   ├── 02_multi_query/
│   ├── 03_parent_doc_retriever/
│   ├── 04_reranking/
│   ├── 05_react_agents/
│   └── 06_langgraph/
└── phase4_expert/              # Weeks 9–12: LLM-as-Judge, RAGAS, LangSmith, Deployment
    ├── 01_llm_as_judge/
    ├── 02_ragas_evaluation/
    ├── 03_langsmith_tracing/
    └── 04_deployment/
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd AgenticAI
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 3. Run any example

```bash
cd phase1_foundation/01_tokens_and_embeddings
python tokens_demo.py
```

### 4. Run tests for any module

```bash
pytest phase1_foundation/01_tokens_and_embeddings/tests/ -v
```

---

## Phase Overview

| Phase | Weeks | Topics | Key Libraries |
|-------|-------|--------|---------------|
| Foundation | 1–3 | Tokens, embeddings, raw APIs, prompt engineering | `anthropic`, `openai`, `tiktoken` |
| Core LangChain + Naive RAG | 3–6 | LCEL chains, PDF loading, chunking, vector DB | `langchain`, `chromadb` |
| Advanced RAG + Agents | 6–9 | HyDE, multi-query, reranking, ReAct, LangGraph | `langchain`, `langgraph`, `cohere` |
| Expert Level | 9–12 | LLM-as-judge, RAGAS, LangSmith tracing, deployment | `ragas`, `langsmith`, `fastapi` |

---

## Prerequisites

- Python 3.11+
- API keys: Anthropic (required), OpenAI (required for embeddings), Cohere (Phase 3+), LangSmith (Phase 4)

## Running All Tests

```bash
pytest --tb=short -q
```

## Security Notes

- Never commit `.env` — it is in `.gitignore`
- All API keys are loaded from environment variables
- Input sanitization is demonstrated in prompt engineering examples
