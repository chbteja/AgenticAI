# Getting Started

Complete guide to setting up and running every example in the curriculum.

---

## Setup (do this once)

```bash
cd /Users/bhanuc/Projects/AgenticAI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Open .env and fill in:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...
```

---

## Phase 1 — Foundation (Weeks 1–3)

### Module 1: Tokens & Embeddings

```bash
cd phase1_foundation/01_tokens_and_embeddings

# tokens_demo.py — works with just ANTHROPIC_API_KEY (tiktoken runs offline)
python tokens_demo.py

# embeddings_demo.py — requires OPENAI_API_KEY
python embeddings_demo.py

# Tests (no API keys needed — all mocked)
pytest tests/test_tokens.py -v
pytest tests/test_embeddings.py -v
```

### Module 2: Raw API

```bash
cd ../02_raw_api

# Requires ANTHROPIC_API_KEY
python anthropic_basics.py

# Requires OPENAI_API_KEY
python openai_basics.py

# Tests
pytest tests/test_anthropic.py -v
pytest tests/test_openai.py -v
```

### Module 3: Prompt Engineering

```bash
cd ../03_prompt_engineering

# Requires ANTHROPIC_API_KEY
python prompting_techniques.py

# Tests
pytest tests/test_prompting.py -v
```

---

## Phase 2 — Core LangChain + Naive RAG (Weeks 3–6)

### Module 1: LCEL Basics

```bash
cd ../../phase2_langchain_rag/01_lcel_basics

# Requires ANTHROPIC_API_KEY
python lcel_chains.py

# Tests
pytest tests/test_lcel.py -v
```

### Module 2: Naive RAG

```bash
cd ../02_naive_rag

# Requires both ANTHROPIC_API_KEY + OPENAI_API_KEY
# First run builds the vector store (takes ~30s for embedding)
python naive_rag.py

# Tests
pytest tests/test_naive_rag.py -v
```

---

## Phase 3 — Advanced RAG + Agents (Weeks 6–9)

### Module 1: HyDE

```bash
cd ../../phase3_advanced_rag_agents/01_hyde

# Requires ANTHROPIC_API_KEY + OPENAI_API_KEY
python hyde_rag.py
pytest tests/test_hyde.py -v
```

### Module 2: Multi-Query

```bash
cd ../02_multi_query

python multi_query_rag.py
pytest tests/test_multi_query.py -v
```

### Module 3: Parent Document Retriever

```bash
cd ../03_parent_doc_retriever

python parent_doc_rag.py
pytest tests/test_parent_doc.py -v
```

### Module 4: Reranking

```bash
cd ../04_reranking

# FlashrankRerank runs locally — no extra API key needed
python reranking_rag.py
pytest tests/test_reranking.py -v
```

### Module 5: ReAct Agents

```bash
cd ../05_react_agents

# Requires ANTHROPIC_API_KEY
python react_agent.py
pytest tests/test_react.py -v
```

### Module 6: LangGraph

```bash
cd ../06_langgraph

# Requires ANTHROPIC_API_KEY
python langgraph_agent.py
pytest tests/test_langgraph.py -v
```

---

## Phase 4 — Expert Level (Weeks 9–12)

### Module 1: LLM-as-Judge

```bash
cd ../../phase4_expert/01_llm_as_judge

# Requires ANTHROPIC_API_KEY (uses claude-sonnet-4-6 as judge)
python llm_judge.py
pytest tests/test_llm_judge.py -v
```

### Module 2: RAGAS Evaluation

```bash
cd ../02_ragas_evaluation

# Requires OPENAI_API_KEY (RAGAS uses OpenAI by default)
python ragas_eval.py
pytest tests/test_ragas.py -v
```

### Module 3: LangSmith Tracing

```bash
cd ../03_langsmith_tracing

# Requires ANTHROPIC_API_KEY; optionally LANGCHAIN_API_KEY for live traces
python langsmith_demo.py
pytest tests/test_langsmith.py -v
```

### Module 4: Deployment (FastAPI)

```bash
cd ../04_deployment

# Start the server (requires ANTHROPIC_API_KEY + OPENAI_API_KEY)
uvicorn fastapi_app:app --reload --port 8000

# In another terminal — test the endpoints
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who created Python?"}'

# Tests (no server needed — uses FastAPI TestClient)
pytest tests/test_deployment.py -v
```

---

## Running All Tests at Once

```bash
cd /Users/bhanuc/Projects/AgenticAI

# All tests across all phases (quiet summary)
pytest --tb=short -q

# All tests with full verbose output
pytest -v

# One phase at a time
pytest phase1_foundation/ -v
pytest phase2_langchain_rag/ -v
pytest phase3_advanced_rag_agents/ -v
pytest phase4_expert/ -v
```

---

## API Key Requirements

| Phase / Module | ANTHROPIC_API_KEY | OPENAI_API_KEY | LANGCHAIN_API_KEY |
|---------------|:-----------------:|:--------------:|:-----------------:|
| Phase 1 — tokens (tiktoken only) | | | |
| Phase 1 — raw API, prompt engineering | ✓ | ✓ | |
| Phase 2–3 — all RAG modules | ✓ | ✓ | |
| Phase 4 — LLM-as-judge | ✓ | | |
| Phase 4 — RAGAS | | ✓ | |
| Phase 4 — LangSmith (optional) | ✓ | | ✓ (optional) |
| Phase 4 — Deployment | ✓ | ✓ | |
| **All test suites** | **none** | **none** | **none** |

> All test suites mock external API calls — you can run the full test suite without any API keys.

---

## Troubleshooting

**`EnvironmentError: ANTHROPIC_API_KEY not set`**
— Check that `.env` exists and contains the key. Make sure you ran `load_dotenv()` or that you sourced the file.

**`ModuleNotFoundError`**
— Run `pip install -r requirements.txt` inside your activated virtual environment.

**`chromadb` errors on first run**
— The vector store is built on the first run and cached in `chroma_db/`. If it gets corrupted, delete the directory and re-run.

**Phase 3+ examples say "Sample data not found"**
— The advanced RAG modules read from `phase2_langchain_rag/02_naive_rag/data/sample.txt`. Run from the project root or ensure that path exists.

**FastAPI `/ready` returns 503**
— The RAG pipeline hasn't been built yet. Send `POST /index` to trigger background indexing, or ensure both API keys are set at server startup.
