# Phase 4 — Expert Level (Weeks 9–12)

## Overview

Production AI systems need more than working code — they need evaluation, observability,
and reliable deployment. This phase covers the tools and patterns that separate
experimental prototypes from production-grade systems.

## Modules

| Module | What You Build | Key Skill |
|--------|---------------|-----------|
| `01_llm_as_judge` | LLM-scored evaluation harness | Automated quality assessment |
| `02_ragas_evaluation` | RAGAS metrics pipeline | RAG evaluation framework |
| `03_langsmith_tracing` | Traced LLM application | Debugging with traces |
| `04_deployment` | FastAPI RAG service | REST API + production patterns |

## Learning Path

1. **Evaluate before you optimise**: build the LLM-as-judge harness first so you can measure improvement
2. **RAGAS**: learn its 4 core metrics and what each one tells you about your RAG pipeline
3. **LangSmith**: trace 10 real queries through your RAG pipeline and identify the biggest quality gap
4. **Deployment**: wrap your Phase 3 RAG chain in a FastAPI app with health checks and rate limiting

## Key Insight

You can't improve what you can't measure. LLM-as-judge and RAGAS give you quantitative
signals that replace manual spot-checking at scale.
