"""
hyde_rag.py — HyDE: Hypothetical Document Embeddings

Purpose:
    Fix a fundamental naive RAG failure: user queries are often short and
    low-information ("what year was Python created?"), while documents contain
    rich answer text. The query and the relevant document fragment live in
    different embedding spaces.

    HyDE solution: Ask the LLM to generate a *hypothetical answer* to the query
    before searching. The hypothetical answer uses similar vocabulary and style
    to the real answer, so its embedding matches the real document better.

    Flow: query → LLM generates hypothetical answer → embed hypothetical answer
          → search vector store → retrieve real docs → LLM answers from real context

Learning Objectives:
    1. Understand why short queries underperform in vector search.
    2. Implement HyDE: generate a hypothetical answer and embed it instead of the query.
    3. Compare HyDE vs naive search on the same questions.
    4. Understand when HyDE helps vs when it doesn't.

Tech Stack: langchain, langchain-anthropic, langchain-openai, langchain-chroma
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

GEN_MODEL = "claude-haiku-4-5-20251001"
EMBED_MODEL = "text-embedding-3-small"
DATA_DIR = Path(__file__).parent.parent / "01_hyde" / "data"
SAMPLE_DATA = Path(__file__).parent.parent.parent / "phase2_langchain_rag/02_naive_rag/data/sample.txt"


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


def _load_and_index(data_path: Path):
    """Load the sample document and build a Chroma vector store (in-memory)."""
    logger.info("_load_and_index: %s", data_path)
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = data_path.read_text(encoding="utf-8")
    docs = [Document(page_content=text, metadata={"source": str(data_path)})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info("Indexed %d chunks", len(chunks))

    embeddings = _get_embeddings()
    vector_store = Chroma.from_documents(chunks, embedding=embeddings)
    return vector_store


def generate_hypothetical_answer(query: str) -> str:
    """
    Ask the LLM to write a hypothetical document that would answer the query.

    This is the core of HyDE. The generated text is NOT shown to the user —
    it is only used as a better search query.

    Args:
        query: The user's original question.

    Returns:
        A paragraph-length hypothetical answer.
    """
    logger.info("generate_hypothetical_answer: query=%r", query[:60])

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(
        """Please write a short paragraph (2-3 sentences) that would be a good answer
to the following question. Write it as if you found it in a reference document.
Do not say 'I' or 'we' — write in a factual, encyclopaedic style.

Question: {query}

Hypothetical answer:"""
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"query": query})
    logger.info("Hypothetical answer generated: %d chars", len(answer))
    logger.debug("Hypothetical answer: %r", answer[:100])
    return answer


def naive_search(query: str, vector_store, k: int = 4) -> list[Document]:
    """Search using the raw user query (baseline naive RAG)."""
    logger.info("naive_search: query=%r, k=%d", query[:60], k)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    logger.info("naive_search: %d docs retrieved", len(docs))
    return docs


def hyde_search(query: str, vector_store, k: int = 4) -> list[Document]:
    """
    Search using a LLM-generated hypothetical answer (HyDE).

    Steps:
        1. Generate hypothetical answer from the LLM.
        2. Embed the hypothetical answer (not the query).
        3. Search the vector store with the hypothetical answer embedding.

    Args:
        query:        Original user query.
        vector_store: Chroma vector store.
        k:            Number of documents to retrieve.

    Returns:
        Retrieved documents, likely more relevant than naive search.
    """
    logger.info("hyde_search: query=%r, k=%d", query[:60], k)

    hypothetical_answer = generate_hypothetical_answer(query)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Search with the hypothetical answer instead of the original query
    docs = retriever.invoke(hypothetical_answer)
    logger.info("hyde_search: %d docs retrieved", len(docs))
    return docs


def build_hyde_chain(vector_store):
    """
    Build a complete HyDE RAG chain.

    Chain flow:
        1. question → generate_hypothetical_answer → hypothetical_doc
        2. hypothetical_doc → vector_store similarity search → retrieved_docs
        3. retrieved_docs + question → LLM → final answer

    Returns:
        A Runnable accepting {"question": str} → str.
    """
    logger.info("build_hyde_chain: building")

    llm = _get_llm()
    parser = StrOutputParser()

    # Step 1: Generate hypothetical document
    hyde_prompt = ChatPromptTemplate.from_template(
        "Write a 2-sentence factual paragraph that would answer: {question}"
    )
    hypothetical_doc_chain = hyde_prompt | llm | parser

    # Step 2: Use hypothetical doc as the search query
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(d.page_content for d in docs)

    # Step 3: Final answer generation
    answer_prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}
Answer:"""
    )

    # Full pipeline
    chain = (
        RunnablePassthrough.assign(
            hypothetical=hypothetical_doc_chain
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["hypothetical"]))
        )
        | answer_prompt
        | llm
        | parser
    )

    logger.info("HyDE chain built")
    return chain


def compare_naive_vs_hyde(query: str, vector_store) -> dict:
    """
    Run the same query through both naive search and HyDE, return comparison.

    Returns:
        Dict with keys: query, naive_docs, hyde_docs, naive_first_chunk, hyde_first_chunk.
    """
    logger.info("compare_naive_vs_hyde: query=%r", query[:60])

    naive_docs = naive_search(query, vector_store)
    hyde_docs = hyde_search(query, vector_store)

    return {
        "query": query,
        "naive_first_chunk": naive_docs[0].page_content[:200] if naive_docs else None,
        "hyde_first_chunk": hyde_docs[0].page_content[:200] if hyde_docs else None,
        "naive_doc_count": len(naive_docs),
        "hyde_doc_count": len(hyde_docs),
    }


def main() -> None:
    logger.info("=== HyDE RAG Demo Starting ===")

    data_path = SAMPLE_DATA
    if not data_path.exists():
        logger.error("Sample data not found at %s — run phase2 examples first", data_path)
        return

    print("\n=== Building Vector Store ===")
    vector_store = _load_and_index(data_path)
    print(f"Vector store ready: {vector_store._collection.count()} chunks")

    test_queries = [
        "Python creator",         # Very short — HyDE should help significantly
        "When was Python released?",
        "web frameworks for Python",
    ]

    print("\n=== HyDE vs Naive Search Comparison ===")
    for query in test_queries:
        print(f"\nQuery: {query!r}")

        # Show hypothetical answer
        hypo = generate_hypothetical_answer(query)
        print(f"Hypothetical answer: {hypo[:150]}...")

        comparison = compare_naive_vs_hyde(query, vector_store)
        print(f"Naive top chunk:  {comparison['naive_first_chunk']}")
        print(f"HyDE  top chunk:  {comparison['hyde_first_chunk']}")

    print("\n=== Full HyDE Chain Q&A ===")
    hyde_chain = build_hyde_chain(vector_store)
    questions = ["Who invented Python?", "What are Python's web frameworks?"]
    for q in questions:
        answer = hyde_chain.invoke({"question": q})
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    logger.info("=== HyDE RAG Demo Complete ===")


if __name__ == "__main__":
    main()
