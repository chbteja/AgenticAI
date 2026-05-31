"""
multi_query_rag.py — Multi-Query Retrieval

Purpose:
    Improve retrieval recall by generating multiple semantically diverse query
    variants and taking the union of retrieved documents across all of them.

Learning Objectives:
    1. Understand how a single query can miss relevant documents.
    2. Generate query variants using an LLM.
    3. Retrieve across all variants and deduplicate results.
    4. Build the full multi-query RAG chain.
    5. Compare recall vs naive single-query retrieval.

Tech Stack: langchain, langchain-anthropic, langchain-openai, langchain-chroma
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
NUM_VARIANTS = 3
TOP_K_PER_QUERY = 3


def _get_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(api_key=api_key, model=GEN_MODEL, max_tokens=256)


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
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    embeddings = _get_embeddings()
    return Chroma.from_documents(chunks, embedding=embeddings)


def generate_query_variants(query: str, n: int = NUM_VARIANTS) -> list[str]:
    """
    Generate *n* semantically diverse rephrasing of *query*.

    Uses an LLM to produce alternative formulations that might match different
    document vocabulary. Diversity is the goal — avoid paraphrases that are
    too similar to each other.

    Args:
        query: Original user query.
        n:     Number of variants to generate.

    Returns:
        List of query variant strings (length n). Does NOT include the original query.
    """
    logger.info("generate_query_variants: query=%r, n=%d", query[:60], n)

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(
        """Generate {n} different versions of the following question.
Each version should use different vocabulary and phrasing, but ask about the same topic.
Return ONLY the questions, one per line, with no numbering or extra text.

Original question: {query}

{n} alternative versions:"""
    )
    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke({"query": query, "n": n})

    # Parse the newline-separated response into a clean list
    variants = [
        line.strip()
        for line in raw_output.splitlines()
        if line.strip() and line.strip() != query
    ][:n]

    logger.info("Generated %d variants: %s", len(variants), variants)
    return variants


def multi_query_retrieve(
    query: str,
    vector_store,
    n_variants: int = NUM_VARIANTS,
    k_per_query: int = TOP_K_PER_QUERY,
) -> list[Document]:
    """
    Retrieve documents using multiple query variants and deduplicate.

    Steps:
        1. Generate n query variants.
        2. Retrieve top-k docs for the original query + all variants.
        3. Deduplicate by page_content hash (exact dedup).

    Args:
        query:       Original user query.
        vector_store: Chroma vector store.
        n_variants:  Number of query variants to generate.
        k_per_query: Documents to retrieve per query.

    Returns:
        Deduplicated union of all retrieved documents.
    """
    logger.info(
        "multi_query_retrieve: query=%r, n_variants=%d, k_per_query=%d",
        query[:60], n_variants, k_per_query,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": k_per_query})

    # Start with the original query
    all_queries = [query] + generate_query_variants(query, n_variants)
    logger.info("Running retrieval for %d queries", len(all_queries))

    seen_content: set[str] = set()
    unique_docs: list[Document] = []

    for q in all_queries:
        docs = retriever.invoke(q)
        logger.debug("Query %r → %d docs", q[:40], len(docs))
        for doc in docs:
            # Use the first 200 chars as a dedup key (fast, good enough)
            key = doc.page_content[:200]
            if key not in seen_content:
                seen_content.add(key)
                unique_docs.append(doc)

    logger.info(
        "multi_query_retrieve: %d unique docs from %d queries",
        len(unique_docs), len(all_queries),
    )
    return unique_docs


def build_multi_query_chain(vector_store):
    """
    Build a complete multi-query RAG chain.

    Returns:
        A Runnable accepting {"question": str} → str.
    """
    logger.info("build_multi_query_chain: building")
    llm = _get_llm()

    def retrieve(inputs: dict) -> dict:
        """Retrieval step — runs multi-query and formats context."""
        docs = multi_query_retrieve(inputs["question"], vector_store)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        return {"context": context, "question": inputs["question"], "doc_count": len(docs)}

    answer_prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the provided context.
If the answer is not in the context, say so clearly.

Context ({doc_count} relevant sections):
{context}

Question: {question}
Answer:"""
    )

    chain = retrieve | answer_prompt | llm | StrOutputParser()
    logger.info("Multi-query chain built")
    return chain


def main() -> None:
    logger.info("=== Multi-Query RAG Demo Starting ===")

    if not SAMPLE_DATA.exists():
        logger.error("Sample data not found: %s", SAMPLE_DATA)
        return

    print("\n=== Building Vector Store ===")
    vector_store = _load_vector_store(SAMPLE_DATA)
    print(f"Vector store ready: {vector_store._collection.count()} chunks")

    # ── Demonstrate query variants ────────────────────────────────────────────
    print("\n=== Query Variant Generation ===")
    original = "Python web development"
    variants = generate_query_variants(original, n=3)
    print(f"Original: {original!r}")
    for i, v in enumerate(variants, 1):
        print(f"Variant {i}: {v!r}")

    # ── Compare naive vs multi-query ──────────────────────────────────────────
    print("\n=== Retrieval Comparison ===")
    test_query = "Python web development frameworks"

    naive_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    naive_docs = naive_retriever.invoke(test_query)
    multi_docs = multi_query_retrieve(test_query, vector_store, n_variants=3, k_per_query=3)

    print(f"Naive retrieval:       {len(naive_docs)} unique documents")
    print(f"Multi-query retrieval: {len(multi_docs)} unique documents")

    # ── Full chain Q&A ────────────────────────────────────────────────────────
    print("\n=== Multi-Query Chain Q&A ===")
    chain = build_multi_query_chain(vector_store)
    questions = [
        "What web frameworks does Python have?",
        "Who governs Python's development?",
    ]
    for q in questions:
        answer = chain.invoke({"question": q})
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    logger.info("=== Multi-Query RAG Demo Complete ===")


if __name__ == "__main__":
    main()
