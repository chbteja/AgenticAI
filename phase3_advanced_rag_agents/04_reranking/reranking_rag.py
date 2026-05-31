"""
reranking_rag.py — Cross-Encoder Reranking

Purpose:
    Improve retrieval precision by reranking an initial set of retrieved documents
    using a cross-encoder model that scores query-document relevance jointly.

    Strategy: over-retrieve (top-20) with a fast bi-encoder, then rerank to top-5
    with a precise cross-encoder. Send only the reranked top-5 to the LLM.

Learning Objectives:
    1. Understand the bi-encoder/cross-encoder trade-off.
    2. Use FlashrankRerank (local, no API key) for reranking.
    3. Optionally use CohereRerank (API-based) for higher quality.
    4. See ranking position changes before and after reranking.

Tech Stack: langchain, langchain-community (FlashrankRerank), langchain-openai, langchain-anthropic
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_DATA = Path(__file__).parent.parent.parent / "phase2_langchain_rag/02_naive_rag/data/sample.txt"
GEN_MODEL = "claude-haiku-4-5-20251001"
EMBED_MODEL = "text-embedding-3-small"
INITIAL_K = 10   # Over-retrieve for reranking
FINAL_K = 3      # Keep after reranking


def _get_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(api_key=api_key, model=GEN_MODEL, max_tokens=512)


def _get_embeddings():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)


def _load_vector_store(data_path: Path):
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = data_path.read_text(encoding="utf-8")
    docs = [Document(page_content=text, metadata={"source": str(data_path)})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    embeddings = _get_embeddings()
    return Chroma.from_documents(chunks, embedding=embeddings)


def rerank_with_flashrank(
    query: str,
    documents: list[Document],
    top_n: int = FINAL_K,
) -> list[Document]:
    """
    Rerank documents using FlashrankRerank (local cross-encoder, no API key needed).

    FlashrankRerank is a LangChain document compressor that uses the flashrank library.
    It runs locally and is fast enough for production use.

    Args:
        query:     The user's query.
        documents: Candidate documents from initial retrieval.
        top_n:     Number of top-ranked documents to return.

    Returns:
        Top-n documents sorted by cross-encoder relevance score.
    """
    logger.info("rerank_with_flashrank: %d docs → top %d", len(documents), top_n)

    try:
        from langchain_community.document_compressors import FlashrankRerank
        compressor = FlashrankRerank(top_n=top_n)
    except ImportError:
        logger.warning("flashrank not installed — returning documents unchanged. Run: pip install flashrank")
        return documents[:top_n]

    reranked = compressor.compress_documents(documents, query)
    logger.info("rerank_with_flashrank: %d docs after reranking", len(reranked))
    return reranked


def rerank_with_cohere(
    query: str,
    documents: list[Document],
    top_n: int = FINAL_K,
) -> list[Document]:
    """
    Rerank documents using Cohere's reranking API.

    Higher quality than FlashrankRerank but requires COHERE_API_KEY and makes API calls.

    Args:
        query:     The user's query.
        documents: Candidate documents from initial retrieval.
        top_n:     Number of top-ranked documents to return.

    Returns:
        Top-n documents sorted by Cohere relevance score.

    Raises:
        EnvironmentError: If COHERE_API_KEY is not set.
    """
    logger.info("rerank_with_cohere: %d docs → top %d", len(documents), top_n)

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY not set — needed for Cohere reranking")

    from langchain_community.document_compressors import CohereRerank

    compressor = CohereRerank(
        cohere_api_key=api_key,
        top_n=top_n,
        model="rerank-english-v3.0",
    )
    reranked = compressor.compress_documents(documents, query)
    logger.info("rerank_with_cohere: %d docs after reranking", len(reranked))
    return reranked


def retrieve_then_rerank(
    query: str,
    vector_store,
    initial_k: int = INITIAL_K,
    final_k: int = FINAL_K,
    use_cohere: bool = False,
) -> dict:
    """
    Full retrieve-then-rerank pipeline with ranking change analysis.

    Steps:
        1. Retrieve top-{initial_k} documents using bi-encoder (fast).
        2. Rerank to top-{final_k} using cross-encoder (precise).
        3. Return both sets for comparison.

    Args:
        query:       User query.
        vector_store: Chroma vector store.
        initial_k:   Over-retrieve this many documents initially.
        final_k:     Keep this many after reranking.
        use_cohere:  Use Cohere API (True) or local Flashrank (False).

    Returns:
        Dict with keys: query, initial_docs, reranked_docs, ranking_changes.
    """
    logger.info("retrieve_then_rerank: k=%d → %d, cohere=%s", initial_k, final_k, use_cohere)

    # ── Step 1: Initial retrieval ──────────────────────────────────────────────
    retriever = vector_store.as_retriever(search_kwargs={"k": initial_k})
    initial_docs = retriever.invoke(query)
    logger.info("Initial retrieval: %d docs", len(initial_docs))

    # ── Step 2: Reranking ──────────────────────────────────────────────────────
    if use_cohere and os.environ.get("COHERE_API_KEY"):
        reranked_docs = rerank_with_cohere(query, initial_docs, top_n=final_k)
    else:
        reranked_docs = rerank_with_flashrank(query, initial_docs, top_n=final_k)

    # ── Step 3: Analyse ranking changes ───────────────────────────────────────
    initial_order = [d.page_content[:80] for d in initial_docs]
    reranked_order = [d.page_content[:80] for d in reranked_docs]

    return {
        "query": query,
        "initial_docs": initial_docs,
        "reranked_docs": reranked_docs,
        "initial_top_snippet": initial_docs[0].page_content[:150] if initial_docs else None,
        "reranked_top_snippet": reranked_docs[0].page_content[:150] if reranked_docs else None,
    }


def build_reranking_rag_chain(vector_store):
    """
    Build a RAG chain with retrieve-then-rerank.

    Returns:
        A Runnable accepting {"question": str} → str.
    """
    logger.info("build_reranking_rag_chain: building")
    llm = _get_llm()

    def retrieve_rerank_format(question: str) -> dict:
        result = retrieve_then_rerank(question, vector_store)
        context = "\n\n---\n\n".join(d.page_content for d in result["reranked_docs"])
        logger.info(
            "Context after reranking: %d docs, %d chars",
            len(result["reranked_docs"]), len(context),
        )
        return {"context": context, "question": question}

    prompt = ChatPromptTemplate.from_template(
        """Answer using ONLY the context provided. Say so if the answer isn't there.

Context (reranked for relevance):
{context}

Question: {question}
Answer:"""
    )

    chain = retrieve_rerank_format | prompt | llm | StrOutputParser()
    logger.info("Reranking RAG chain built")
    return chain


def main() -> None:
    logger.info("=== Reranking RAG Demo Starting ===")

    if not SAMPLE_DATA.exists():
        logger.error("Sample data not found: %s", SAMPLE_DATA)
        return

    print("\n=== Building Vector Store ===")
    vector_store = _load_vector_store(SAMPLE_DATA)
    print(f"Vector store ready: {vector_store._collection.count()} chunks")

    # ── Show ranking changes ───────────────────────────────────────────────────
    print("\n=== Ranking Comparison ===")
    query = "Python web frameworks Flask Django"
    result = retrieve_then_rerank(query, vector_store, initial_k=8, final_k=3)
    print(f"Query: {query!r}")
    print(f"\nInitial top doc:\n  {result['initial_top_snippet']}")
    print(f"\nReranked top doc:\n  {result['reranked_top_snippet']}")

    print("\nInitial ranking (first 3):")
    for i, doc in enumerate(result["initial_docs"][:3], 1):
        print(f"  [{i}] {doc.page_content[:80]}...")

    print("\nAfter reranking (top 3):")
    for i, doc in enumerate(result["reranked_docs"][:3], 1):
        print(f"  [{i}] {doc.page_content[:80]}...")

    # ── Full chain Q&A ─────────────────────────────────────────────────────────
    print("\n=== Reranking RAG Q&A ===")
    chain = build_reranking_rag_chain(vector_store)
    questions = ["What web frameworks does Python have?", "Who created Python?"]
    for q in questions:
        answer = chain.invoke({"question": q})
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    logger.info("=== Reranking RAG Demo Complete ===")


if __name__ == "__main__":
    main()
