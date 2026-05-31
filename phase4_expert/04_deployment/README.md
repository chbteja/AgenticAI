# 04 — Deployment with FastAPI

## Overview

Wrap your RAG pipeline in a production-ready FastAPI REST service with:
- Pydantic request/response models with validation
- Health check endpoint for load balancer probes
- Simple in-memory rate limiting
- Structured logging with request IDs
- Background indexing task

## Learning Objectives

- Expose a RAG chain as a REST API
- Validate requests with Pydantic v2 models
- Implement health checks and readiness probes
- Add basic rate limiting
- Run async endpoints with background tasks

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥0.115 | REST framework |
| `uvicorn` | ≥0.30 | ASGI server |
| `pydantic` | ≥2.0 | Request/response validation |
| `httpx` | ≥0.27 | Async test client |

## How to Run

```bash
# Start the server
uvicorn fastapi_app:app --reload --port 8000

# Query the API
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who created Python?"}'

# Health check
curl http://localhost:8000/health
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (always 200) |
| GET | `/ready` | Readiness (200 if vector store loaded) |
| POST | `/ask` | Ask a question, get an answer |
| POST | `/index` | Trigger background document re-indexing |

## How to Run Tests

```bash
pytest tests/ -v
```

## Production Checklist

Before deploying to production, also add:
- [ ] Authentication (API key or JWT)
- [ ] Persistent rate limiting (Redis)
- [ ] HTTPS termination (nginx or load balancer)
- [ ] Horizontal scaling (each instance can share a Chroma server or use Pinecone)
- [ ] Structured JSON logging → cloud log aggregation
