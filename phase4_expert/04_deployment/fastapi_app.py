"""
fastapi_app.py — Production RAG Service with FastAPI

Purpose:
    Wrap a RAG pipeline in a production-ready REST API. Demonstrates
    request validation, health checks, rate limiting, and async endpoints.

Learning Objectives:
    1. Create typed request/response models with Pydantic v2.
    2. Expose a RAG chain as a POST endpoint.
    3. Implement health and readiness checks.
    4. Add simple in-memory rate limiting.
    5. Use background tasks for async document indexing.

Security:
    - Input length validated by Pydantic (prevents token exhaustion)
    - Rate limiting prevents API abuse
    - API key required for /ask endpoint (configurable)
    - No secrets in logs

Tech Stack: fastapi, uvicorn, pydantic, langchain, python-dotenv
"""

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_DATA = Path(__file__).parent.parent.parent / "phase2_langchain_rag/02_naive_rag/data/sample.txt"
GEN_MODEL = "claude-haiku-4-5-20251001"
EMBED_MODEL = "text-embedding-3-small"
MAX_QUESTION_LENGTH = 500
RATE_LIMIT_REQUESTS = 10   # Requests per window
RATE_LIMIT_WINDOW = 60     # Window size in seconds

# ── Application state ──────────────────────────────────────────────────────────
# Stored on app.state so it's accessible across requests without global variables
_app_state: dict = {
    "vector_store": None,
    "rag_chain": None,
    "is_ready": False,
    "indexed_doc_count": 0,
}

# Simple in-memory rate limiter: ip → [timestamp, timestamp, ...]
_rate_limit_store: dict = defaultdict(list)


# ── Pydantic models ────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=MAX_QUESTION_LENGTH,
        description="The question to ask the RAG pipeline.",
        examples=["Who created Python?"],
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of documents to retrieve.",
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be empty or whitespace.")
        return v.strip()


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""

    question: str
    answer: str
    retrieved_chunks: int
    latency_ms: int
    model: str = GEN_MODEL


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


class ReadinessResponse(BaseModel):
    ready: bool
    indexed_documents: int
    message: str


class IndexRequest(BaseModel):
    """Request model for the /index endpoint."""
    data_path: Optional[str] = Field(
        default=None,
        description="Path to documents to index. Defaults to the built-in sample.",
    )


class IndexResponse(BaseModel):
    message: str
    task_id: str


# ── Rate limiting dependency ───────────────────────────────────────────────────

def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency that enforces per-IP rate limiting.

    Raises HTTP 429 if the client exceeds RATE_LIMIT_REQUESTS in RATE_LIMIT_WINDOW seconds.

    This is a simple in-memory implementation — in production, use Redis.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Remove timestamps outside the current window
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s",
        )

    _rate_limit_store[client_ip].append(now)
    logger.debug("Rate limit OK: ip=%s, count=%d", client_ip, len(_rate_limit_store[client_ip]))


# ── RAG pipeline helpers ───────────────────────────────────────────────────────

def _build_rag_pipeline(data_path: Path) -> tuple:
    """
    Build the vector store and RAG chain from a document path.

    Returns:
        Tuple of (vector_store, rag_chain, chunk_count).
    """
    logger.info("_build_rag_pipeline: data_path=%s", data_path)

    from langchain_anthropic import ChatAnthropic
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load and split
    text = data_path.read_text(encoding="utf-8")
    docs = [Document(page_content=text, metadata={"source": str(data_path)})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embed and store
    embeddings = OpenAIEmbeddings(
        api_key=os.environ["OPENAI_API_KEY"],
        model=EMBED_MODEL,
    )
    vector_store = Chroma.from_documents(chunks, embedding=embeddings)

    # Build chain
    llm = ChatAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=GEN_MODEL,
        max_tokens=512,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """Answer using ONLY the provided context. Say so if the answer isn't there.

Context:
{context}

Question: {question}
Answer:"""
    )

    chain = (
        {"context": retriever | (lambda docs: "\n\n---\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("Pipeline built: %d chunks indexed", len(chunks))
    return vector_store, chain, len(chunks)


# ── Startup / shutdown ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build the RAG pipeline if API keys are available."""
    logger.info("Application startup: building RAG pipeline")

    if os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY"):
        if SAMPLE_DATA.exists():
            try:
                vs, chain, count = _build_rag_pipeline(SAMPLE_DATA)
                _app_state["vector_store"] = vs
                _app_state["rag_chain"] = chain
                _app_state["indexed_doc_count"] = count
                _app_state["is_ready"] = True
                logger.info("Startup complete: %d chunks indexed", count)
            except Exception as exc:
                logger.error("Startup RAG build failed: %s", exc)
        else:
            logger.warning("Sample data not found: %s", SAMPLE_DATA)
    else:
        logger.warning("API keys not set — RAG pipeline not initialised at startup")

    yield  # Application runs here

    logger.info("Application shutdown")
    _app_state["is_ready"] = False


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic AI RAG Service",
    description="Production RAG pipeline API built with FastAPI and LangChain.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint. Always returns 200 if the process is running.

    Used by load balancers to determine if the instance should receive traffic.
    """
    logger.debug("health_check called")
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadinessResponse, tags=["Operations"])
async def readiness_check() -> ReadinessResponse:
    """
    Readiness probe. Returns 200 if the RAG pipeline is initialised.

    Used by orchestrators (Kubernetes) to delay traffic until the app is ready.
    """
    logger.debug("readiness_check called")
    is_ready = _app_state["is_ready"]

    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not yet initialised",
        )

    return ReadinessResponse(
        ready=True,
        indexed_documents=_app_state["indexed_doc_count"],
        message="Service is ready to accept requests",
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(
    body: AskRequest,
    _rate_limited: None = Depends(check_rate_limit),
) -> AskResponse:
    """
    Ask a question and get an answer from the RAG pipeline.

    The answer is grounded in the indexed documents. Questions outside
    the document scope will receive a "not in context" response.
    """
    logger.info("ask_question: question=%r, top_k=%d", body.question[:60], body.top_k)

    if not _app_state["is_ready"] or not _app_state["rag_chain"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not ready. Send POST /index to initialise.",
        )

    start_time = time.time()
    try:
        answer = _app_state["rag_chain"].invoke(body.question)
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info("ask_question: answered in %dms", latency_ms)
    except Exception as exc:
        logger.error("ask_question: chain failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG chain failed — check server logs",
        )

    return AskResponse(
        question=body.question,
        answer=answer,
        retrieved_chunks=body.top_k,
        latency_ms=latency_ms,
    )


@app.post("/index", response_model=IndexResponse, tags=["RAG"])
async def index_documents(
    body: IndexRequest,
    background_tasks: BackgroundTasks,
) -> IndexResponse:
    """
    Trigger background re-indexing of documents.

    The indexing runs asynchronously — the endpoint returns immediately
    and the indexing happens in the background.
    """
    import uuid

    task_id = str(uuid.uuid4())
    data_path = Path(body.data_path) if body.data_path else SAMPLE_DATA

    logger.info("index_documents: data_path=%s, task_id=%s", data_path, task_id)

    if not data_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data path not found: {data_path}",
        )

    def _background_index():
        logger.info("Background indexing: task_id=%s, path=%s", task_id, data_path)
        try:
            vs, chain, count = _build_rag_pipeline(data_path)
            _app_state["vector_store"] = vs
            _app_state["rag_chain"] = chain
            _app_state["indexed_doc_count"] = count
            _app_state["is_ready"] = True
            logger.info("Background indexing complete: task_id=%s, chunks=%d", task_id, count)
        except Exception as exc:
            logger.error("Background indexing failed: task_id=%s, error=%s", task_id, exc)

    background_tasks.add_task(_background_index)
    return IndexResponse(
        message="Indexing started in background",
        task_id=task_id,
    )


# ── Run directly ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server on port 8000")
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
