"""
naive_rag.py — Complete Naive RAG Pipeline

Purpose:
    Build a full Retrieval-Augmented Generation (RAG) pipeline from scratch.
    Load documents, split into chunks, embed, store in a vector database,
    retrieve relevant chunks at query time, and generate an answer.

Learning Objectives:
    1. Load and split text documents with RecursiveCharacterTextSplitter.
    2. Generate embeddings with OpenAI and store in Chroma.
    3. Retrieve top-k semantically relevant chunks for a query.
    4. Pass retrieved context to an LLM and generate a grounded answer.
    5. Identify where naive RAG fails and why.

Architecture: Query → Embed → Similarity Search → LLM + Context → Answer

Tech Stack: langchain, langchain-openai, langchain-anthropic, langchain-chroma, chromadb
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
# These are the parameters that affect RAG quality most. Change them to experiment.
CHUNK_SIZE = 500      # Characters per chunk — larger = more context, less precision
CHUNK_OVERLAP = 50    # Characters of overlap between chunks — prevents split sentences
TOP_K = 4             # Number of chunks to retrieve per query
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "claude-haiku-4-5-20251001"

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_embeddings():
    """Create OpenAI embeddings model. Requires OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set — needed for embedding generation")

    from langchain_openai import OpenAIEmbeddings
    logger.debug("Creating OpenAI embeddings: model=%s", EMBED_MODEL)
    return OpenAIEmbeddings(api_key=api_key, model=EMBED_MODEL)


def _get_llm():
    """Create Anthropic chat model. Requires ANTHROPIC_API_KEY."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — needed for answer generation")

    from langchain_anthropic import ChatAnthropic
    logger.debug("Creating Anthropic LLM: model=%s", GEN_MODEL)
    return ChatAnthropic(api_key=api_key, model=GEN_MODEL, max_tokens=1024)


# ── Document loading and splitting ────────────────────────────────────────────

def load_documents(source_path: Path) -> list[Document]:
    """
    Load documents from a file or directory.

    Supports .txt and .pdf files. PDFs require pypdf to be installed.

    Args:
        source_path: Path to a file or directory of files.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If source_path does not exist.
        ValueError: If no supported files are found.
    """
    logger.info("load_documents: path=%s", source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {source_path}")

    documents: list[Document] = []

    if source_path.is_file():
        files = [source_path]
    else:
        files = list(source_path.glob("*.txt")) + list(source_path.glob("*.pdf"))

    if not files:
        raise ValueError(f"No .txt or .pdf files found in {source_path}")

    for file_path in files:
        if file_path.suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
            documents.append(Document(page_content=text, metadata={"source": str(file_path)}))
            logger.info("Loaded text file: %s (%d chars)", file_path.name, len(text))

        elif file_path.suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            documents.extend(docs)
            logger.info("Loaded PDF: %s (%d pages)", file_path.name, len(docs))

    logger.info("Total documents loaded: %d", len(documents))
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    The splitter tries to split on paragraph breaks (\\n\\n) first, then sentence
    boundaries (. ? !), then spaces, then characters — in order of preference.
    This preserves semantic units better than a fixed-size split.

    Args:
        documents:     LangChain Document list.
        chunk_size:    Target character count per chunk.
        chunk_overlap: Characters shared between adjacent chunks to prevent context loss.

    Returns:
        List of chunk Documents, each with source metadata preserved.
    """
    logger.info(
        "split_documents: %d docs, chunk_size=%d, overlap=%d",
        len(documents),
        chunk_size,
        chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks (avg %.0f chars)", len(chunks), sum(len(c.page_content) for c in chunks) / max(len(chunks), 1))
    return chunks


# ── Vector store ───────────────────────────────────────────────────────────────

def build_vector_store(
    chunks: list[Document],
    persist_dir: Optional[Path] = CHROMA_DIR,
):
    """
    Embed all chunks and store in a Chroma vector database.

    If persist_dir is provided, the database is saved to disk so you don't
    need to re-embed on every run (embeddings are the most expensive step).

    Args:
        chunks:      List of Document chunks to embed.
        persist_dir: Directory to persist the Chroma database. None = in-memory.

    Returns:
        Chroma vector store instance.
    """
    logger.info("build_vector_store: %d chunks, persist_dir=%s", len(chunks), persist_dir)

    from langchain_chroma import Chroma

    embeddings = _get_embeddings()
    kwargs = {"documents": chunks, "embedding": embeddings}
    if persist_dir:
        persist_dir.mkdir(parents=True, exist_ok=True)
        kwargs["persist_directory"] = str(persist_dir)

    vector_store = Chroma.from_documents(**kwargs)
    logger.info("Vector store built: %d vectors stored", vector_store._collection.count())
    return vector_store


def load_vector_store(persist_dir: Path = CHROMA_DIR):
    """
    Load an existing persisted Chroma vector store from disk.

    Avoids re-embedding when the documents haven't changed.

    Raises:
        FileNotFoundError: If the persist_dir does not exist.
    """
    logger.info("load_vector_store: dir=%s", persist_dir)

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No vector store at {persist_dir}. Run build_vector_store() first."
        )

    from langchain_chroma import Chroma

    embeddings = _get_embeddings()
    vector_store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    count = vector_store._collection.count()
    logger.info("Loaded vector store: %d vectors", count)
    return vector_store


# ── RAG chain ──────────────────────────────────────────────────────────────────

def build_rag_chain(vector_store, top_k: int = TOP_K):
    """
    Build the full RAG Q&A chain.

    Chain flow:
        query → retriever.get_relevant_documents(query) → format_docs() →
        prompt template fills {context} + {question} → LLM → StrOutputParser → answer

    Args:
        vector_store: A LangChain-compatible vector store with .as_retriever().
        top_k:        Number of documents to retrieve per query.

    Returns:
        A Runnable chain that accepts {"question": str} and returns str (the answer).
    """
    logger.info("build_rag_chain: top_k=%d", top_k)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # System prompt for RAG — clear instructions prevent hallucination
    rag_prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer the question, say so clearly.
Do not use your general knowledge — only the provided context.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs: list[Document]) -> str:
        """Concatenate retrieved doc contents with separators."""
        formatted = "\n\n---\n\n".join(doc.page_content for doc in docs)
        logger.debug("format_docs: %d docs → %d chars", len(docs), len(formatted))
        return formatted

    llm = _get_llm()
    parser = StrOutputParser()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | parser
    )

    logger.info("RAG chain built")
    return chain


def retrieve_and_show(
    query: str,
    vector_store,
    top_k: int = TOP_K,
) -> list[Document]:
    """
    Retrieve top-k documents for a query and log them for inspection.

    Use this for debugging — see what the LLM is actually seeing as context.

    Returns:
        List of retrieved Document objects.
    """
    logger.info("retrieve_and_show: query=%r, k=%d", query[:60], top_k)

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    print(f"\n  Retrieved {len(docs)} chunks for query: {query!r}")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        print(f"\n  [{i+1}] Source: {Path(source).name if source != 'unknown' else source}")
        print(f"       Content: {doc.page_content[:150]}...")

    return docs


# ── Demo ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Build the RAG pipeline and answer questions about the sample document."""
    logger.info("=== Naive RAG Demo Starting ===")

    # ── 1. Load and split documents ────────────────────────────────────────────
    print("\n=== 1. Loading and Splitting Documents ===")
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} document(s) → {len(chunks)} chunks")

    # ── 2. Build vector store ──────────────────────────────────────────────────
    print("\n=== 2. Building Vector Store ===")
    vector_store = build_vector_store(chunks)
    print(f"Vector store ready with {vector_store._collection.count()} vectors")

    # ── 3. Build RAG chain ─────────────────────────────────────────────────────
    print("\n=== 3. Building RAG Chain ===")
    rag_chain = build_rag_chain(vector_store)
    print("RAG chain ready")

    # ── 4. Q&A session ────────────────────────────────────────────────────────
    questions = [
        "Who created Python and when?",
        "What are the main Python web frameworks?",
        "What is the Zen of Python?",
        "What happened to Python 2?",
        "What is the capital of France?",  # Out-of-context question — should say "not in context"
    ]

    print("\n=== 4. Q&A Session ===")
    for question in questions:
        print(f"\nQ: {question}")
        logger.info("Invoking RAG chain: question=%r", question)

        # Show retrieved chunks before answering
        retrieve_and_show(question, vector_store)

        # Generate answer
        answer = rag_chain.invoke(question)
        print(f"A: {answer}")

    logger.info("=== Naive RAG Demo Complete ===")


if __name__ == "__main__":
    main()
