# 03 — LangSmith Tracing

## Overview

LangSmith is Anthropic/LangChain's observability platform. It captures every LLM call,
tool invocation, and chain step as a trace — giving you a debuggable timeline of what
happened inside your application.

Without tracing, debugging an agent or RAG pipeline that gives a wrong answer requires
reading code and guessing. With tracing, you can see exactly which retrieved chunks were
passed to the LLM and exactly what it said.

## Learning Objectives

- Configure LangSmith tracing with environment variables
- Name traces and runs for easy filtering in the UI
- Log custom metadata (feedback, expected answers) to runs
- Use the LangSmith SDK to programmatically query trace data
- Understand the relationship between projects, runs, and feedback

## Setup

```bash
# Set in .env
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-ai-curriculum
```

## How to Run

```bash
python langsmith_demo.py

# Tests (mocked — no LangSmith account needed)
pytest tests/ -v
```

## Key Concepts

- **Trace**: One end-to-end request through your system (question → answer)
- **Run**: A single node in the trace (one LLM call, one tool call, one chain step)
- **Feedback**: A score you attach to a run (human rating, automated eval score)
- **Project**: A named collection of traces — use one per experiment or deployment

## Exercises

1. Run 5 questions through the RAG pipeline with tracing enabled, then find the slowest LLM call in the UI.
2. Add a feedback score from the LLM-as-judge module to each trace.
3. Compare traces from naive RAG vs HyDE — observe how the retrieved context differs.
