"""
langsmith_demo.py — LangSmith Tracing and Observability

Purpose:
    Show how to configure LangSmith tracing for a RAG pipeline and how to
    log feedback, custom metadata, and evaluation scores to runs.

Learning Objectives:
    1. Enable LangSmith tracing via environment variables.
    2. Name chains and runs for easy filtering in the UI.
    3. Log feedback (quality scores) to individual runs.
    4. Query recent runs programmatically.
    5. Understand how tracing helps debug agent failures.

Tech Stack: langsmith, langchain, langchain-anthropic, python-dotenv
"""

import logging
import os
import time
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

GEN_MODEL = "claude-haiku-4-5-20251001"


def _is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is configured."""
    return bool(
        os.environ.get("LANGCHAIN_API_KEY")
        and os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )


def setup_tracing(project_name: str = "agentic-ai-curriculum") -> None:
    """
    Configure LangSmith tracing for the current session.

    Tracing is activated by setting environment variables. Once set,
    ALL LangChain calls in this process are automatically traced.

    Args:
        project_name: Name of the LangSmith project to log to.

    Raises:
        EnvironmentError: If LANGCHAIN_API_KEY is not set.
    """
    logger.info("setup_tracing: project=%s", project_name)

    if not os.environ.get("LANGCHAIN_API_KEY"):
        logger.warning(
            "LANGCHAIN_API_KEY not set — tracing is disabled. "
            "Set it in .env to enable LangSmith observability."
        )
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    logger.info("LangSmith tracing enabled: project=%s", project_name)


def build_traced_rag_chain(vector_store, chain_name: str = "rag-demo"):
    """
    Build a RAG chain with LangSmith run names set for easy filtering in the UI.

    Setting run names makes traces much easier to find and correlate.

    Args:
        vector_store: Chroma vector store.
        chain_name:   Name shown in the LangSmith trace UI.

    Returns:
        A Runnable accepting {"question": str} → str.
    """
    logger.info("build_traced_rag_chain: name=%s", chain_name)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    llm = ChatAnthropic(api_key=api_key, model=GEN_MODEL, max_tokens=512)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template(
        """Answer using ONLY the context. Say so if the answer isn't there.

Context:
{context}

Question: {question}
Answer:"""
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Add a human-readable name so this chain is easy to find in LangSmith
    return chain.with_config({"run_name": chain_name})


def log_feedback_to_run(
    run_id: str,
    score: float,
    feedback_key: str = "quality",
    comment: Optional[str] = None,
) -> None:
    """
    Log a quality score to a specific LangSmith run.

    Use this to attach LLM-as-judge scores or human ratings to traces.
    This makes it possible to filter and sort traces by quality score in the UI.

    Args:
        run_id:       UUID of the LangSmith run to score.
        score:        Float score (0–1 for binary, 1–5 for rating scales).
        feedback_key: The dimension name (e.g. "faithfulness", "quality").
        comment:      Optional text explanation for the score.
    """
    logger.info("log_feedback_to_run: run_id=%s, key=%s, score=%.3f", run_id, feedback_key, score)

    if not _is_tracing_enabled():
        logger.warning("Tracing not enabled — feedback not logged to LangSmith")
        return

    try:
        from langsmith import Client
        client = Client()
        client.create_feedback(
            run_id=run_id,
            key=feedback_key,
            score=score,
            comment=comment,
        )
        logger.info("Feedback logged: run_id=%s, key=%s, score=%.3f", run_id, feedback_key, score)
    except Exception as exc:
        logger.error("Failed to log feedback: %s", exc)


def get_recent_runs(
    project_name: str = "agentic-ai-curriculum",
    limit: int = 10,
) -> list[dict]:
    """
    Query recent runs from LangSmith for a project.

    Args:
        project_name: The LangSmith project name.
        limit:        Max number of runs to return.

    Returns:
        List of run summary dicts.

    Raises:
        EnvironmentError: If LangSmith is not configured.
    """
    logger.info("get_recent_runs: project=%s, limit=%d", project_name, limit)

    if not _is_tracing_enabled():
        raise EnvironmentError("LangSmith tracing not configured")

    from langsmith import Client

    client = Client()
    runs = list(client.list_runs(project_name=project_name, limit=limit))

    summaries = []
    for run in runs:
        summaries.append({
            "id": str(run.id),
            "name": run.name,
            "status": run.status,
            "start_time": str(run.start_time),
            "latency_seconds": (
                (run.end_time - run.start_time).total_seconds()
                if run.end_time and run.start_time
                else None
            ),
            "total_tokens": getattr(run, "total_tokens", None),
        })

    logger.info("get_recent_runs: %d runs retrieved", len(summaries))
    return summaries


def run_with_tracing(
    question: str,
    chain,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Run a chain with optional custom metadata attached to the trace.

    Args:
        question: The user's question.
        chain:    The chain to run.
        metadata: Optional dict of metadata to attach to the trace (e.g. user_id, session_id).

    Returns:
        Dict with: answer, run_id (for feedback logging), latency_ms.
    """
    logger.info("run_with_tracing: question=%r", question[:60])

    run_id = str(uuid4())
    config = {"run_id": run_id}
    if metadata:
        config["metadata"] = metadata

    start_time = time.time()
    answer = chain.invoke(question, config=config)
    latency_ms = int((time.time() - start_time) * 1000)

    logger.info("run_with_tracing: latency=%dms, run_id=%s", latency_ms, run_id)
    return {
        "question": question,
        "answer": answer,
        "run_id": run_id,
        "latency_ms": latency_ms,
    }


def demonstrate_tracing_patterns() -> None:
    """
    Show all tracing patterns without requiring a real vector store.

    Uses a simplified chain to demonstrate the tracing API.
    """
    logger.info("demonstrate_tracing_patterns: starting")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[!] ANTHROPIC_API_KEY not set — showing tracing concepts only")
        print("\nTo enable tracing:")
        print("  1. Set LANGCHAIN_API_KEY in .env")
        print("  2. Set LANGCHAIN_TRACING_V2=true in .env")
        print("  3. Set LANGCHAIN_PROJECT=agentic-ai-curriculum in .env")
        print("\nThen every LangChain call will be traced automatically.")
        return

    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatAnthropic(api_key=api_key, model=GEN_MODEL, max_tokens=256)
    chain = (
        ChatPromptTemplate.from_template("Answer in one sentence: {question}")
        | llm
        | StrOutputParser()
    ).with_config({"run_name": "demo-qa-chain"})

    # ── Run with metadata ──────────────────────────────────────────────────────
    result = run_with_tracing(
        question="What is LangSmith used for?",
        chain=chain,
        metadata={"user_id": "student-001", "session": "curriculum-demo"},
    )
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Run ID:  {result['run_id']}")

    # ── Log feedback ───────────────────────────────────────────────────────────
    if _is_tracing_enabled():
        log_feedback_to_run(
            run_id=result["run_id"],
            score=0.9,
            feedback_key="quality",
            comment="Good concise answer.",
        )
        print("Feedback logged to LangSmith")
    else:
        print("(Tracing not enabled — feedback not logged. Configure LANGCHAIN_API_KEY to enable.)")


def main() -> None:
    logger.info("=== LangSmith Tracing Demo Starting ===")

    # ── 1. Setup ───────────────────────────────────────────────────────────────
    print("\n=== 1. Tracing Setup ===")
    setup_tracing()
    enabled = _is_tracing_enabled()
    print(f"Tracing enabled: {enabled}")
    if not enabled:
        print("Set LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true in .env to enable")

    # ── 2. Run with tracing ────────────────────────────────────────────────────
    print("\n=== 2. Run with Tracing ===")
    demonstrate_tracing_patterns()

    # ── 3. Query recent runs ───────────────────────────────────────────────────
    print("\n=== 3. Recent Runs ===")
    if enabled:
        runs = get_recent_runs()
        print(f"Retrieved {len(runs)} recent runs:")
        for run in runs[:5]:
            print(f"  [{run['status']}] {run['name']} — {run['latency_seconds']:.1f}s")
    else:
        print("(LangSmith not configured — no runs to show)")

    logger.info("=== LangSmith Tracing Demo Complete ===")


if __name__ == "__main__":
    main()
