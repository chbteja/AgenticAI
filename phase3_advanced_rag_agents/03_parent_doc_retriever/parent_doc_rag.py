"""
parent_doc_rag.py — Parent Document Retriever

Purpose:
    Solve the chunk-size dilemma by indexing small chunks for precise retrieval
    but returning larger parent chunks to the LLM for rich context.

Learning Objectives:
    1. Understand the trade-off between chunk size and retrieval quality.
    2. Set up a ParentDocumentRetriever with child and parent splitters.
    3. Use InMemoryStore to hold parent documents.
    4. Compare responses from small-chunk vs parent-doc retrieval.

Architecture:
    - Child chunks (200 chars): stored in vector DB, used for retrieval matching
    - Parent chunks (1000 chars): stored in InMemoryStore, returned to LLM

Tech Stack: langchain, langchain-community, langchain-openai, langchain-anthropic
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_DATA = Path(__file__).parent.parent.parent / "phase2_langchain_rag/02_naive_rag/data/sample.txt"
GEN_MODEL = "claude-haiku-4-5-20251001"
EMBED_MODEL = "text-embedding-3-small"


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


def build_parent_doc_retriever(documents: list[Document]):
    """
    Build a ParentDocumentRetriever with:
    - child_splitter: 200-char chunks stored in vector store (for retrieval)
    - parent_splitter: 1000-char chunks stored in InMemoryStore (sent to LLM)

    Args:
        documents: List of Document objects to index.

    Returns:
        ParentDocumentRetriever ready to use with .invoke(query).
    """
    logger.info("build_parent_doc_retriever: %d documents", len(documents))

    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    from langchain_chroma import Chroma

    # Small chunks for precise vector search
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )

    # Larger parent chunks returned to the LLM for context
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    embeddings = _get_embeddings()
    vector_store = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
    )

    # InMemoryStore holds the parent chunks keyed by ID
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Index the documents (creates both child and parent chunks)
    retriever.add_documents(documents)

    # Count child chunks in vector store
    child_count = vector_store._collection.count()
    logger.info(
        "Indexed: %d child chunks in vector store, parents in InMemoryStore",
        child_count,
    )
    return retriever


def compare_chunk_sizes(query: str, documents: list[Document]) -> dict:
    """
    Compare what the LLM receives with small chunks vs parent chunks.

    Returns dict with keys: query, small_chunk_context_len, parent_context_len,
    small_first_chunk, parent_first_chunk.
    """
    logger.info("compare_chunk_sizes: query=%r", query[:60])

    embeddings = _get_embeddings()

    # ── Small chunk retriever (naive) ──────────────────────────────────────────
    from langchain_chroma import Chroma

    small_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    small_chunks = small_splitter.split_documents(documents)
    small_vs = Chroma.from_documents(small_chunks, embedding=embeddings)
    small_retriever = small_vs.as_retriever(search_kwargs={"k": 3})
    small_docs = small_retriever.invoke(query)

    # ── Parent doc retriever ───────────────────────────────────────────────────
    parent_retriever = build_parent_doc_retriever(documents)
    parent_docs = parent_retriever.invoke(query)

    small_total = sum(len(d.page_content) for d in small_docs)
    parent_total = sum(len(d.page_content) for d in parent_docs)

    return {
        "query": query,
        "small_chunk_context_len": small_total,
        "parent_context_len": parent_total,
        "small_first_chunk": small_docs[0].page_content if small_docs else None,
        "parent_first_chunk": parent_docs[0].page_content if parent_docs else None,
    }


def build_parent_rag_chain(documents: list[Document]):
    """
    Build a RAG chain using the ParentDocumentRetriever.

    Returns:
        A Runnable accepting {"question": str} → str.
    """
    logger.info("build_parent_rag_chain: building")

    retriever = build_parent_doc_retriever(documents)
    llm = _get_llm()

    def retrieve_and_format(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        logger.info("Retrieved %d parent docs (%d chars total)", len(docs), len(context))
        return {"context": context, "question": question}

    prompt = ChatPromptTemplate.from_template(
        """Answer using ONLY the context. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}
Answer:"""
    )

    chain = retrieve_and_format | prompt | llm | StrOutputParser()
    logger.info("Parent doc RAG chain built")
    return chain


def main() -> None:
    logger.info("=== Parent Document Retriever Demo Starting ===")

    if not SAMPLE_DATA.exists():
        logger.error("Sample data not found: %s", SAMPLE_DATA)
        return

    text = SAMPLE_DATA.read_text(encoding="utf-8")
    documents = [Document(page_content=text, metadata={"source": str(SAMPLE_DATA)})]

    # ── Chunk size comparison ──────────────────────────────────────────────────
    print("\n=== Chunk Size Comparison ===")
    query = "What are Python's main features and philosophy?"
    comparison = compare_chunk_sizes(query, documents)
    print(f"Query: {query!r}")
    print(f"Small chunks context:  {comparison['small_chunk_context_len']} chars")
    print(f"Parent chunks context: {comparison['parent_context_len']} chars")
    print(f"\nSmall first chunk:\n{comparison['small_first_chunk']}")
    print(f"\nParent first chunk (first 300 chars):\n{comparison['parent_first_chunk'][:300] if comparison['parent_first_chunk'] else 'None'}...")

    # ── Full chain Q&A ─────────────────────────────────────────────────────────
    print("\n=== Parent Document RAG Q&A ===")
    chain = build_parent_rag_chain(documents)
    questions = ["What is the Zen of Python?", "How is Python governed?"]
    for q in questions:
        answer = chain.invoke({"question": q})
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    logger.info("=== Parent Document Retriever Demo Complete ===")


if __name__ == "__main__":
    main()
