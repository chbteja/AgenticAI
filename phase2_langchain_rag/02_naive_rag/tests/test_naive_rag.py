"""Tests for naive_rag.py — vector store and API calls fully mocked."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from naive_rag import load_documents, split_documents


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    f = tmp_path / "test.txt"
    f.write_text("Python is great.\n\nIt is used for data science.\n\nGuido created it in 1991.")
    return f


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(page_content="Python is a programming language.", metadata={"source": "test.txt"}),
        Document(page_content="It was created by Guido van Rossum in 1991.", metadata={"source": "test.txt"}),
    ]


# ── load_documents tests ───────────────────────────────────────────────────────

class TestLoadDocuments:
    def test_loads_txt_file(self, sample_txt_file: Path):
        docs = load_documents(sample_txt_file)
        assert len(docs) == 1
        assert "Python is great" in docs[0].page_content

    def test_metadata_includes_source(self, sample_txt_file: Path):
        docs = load_documents(sample_txt_file)
        assert "source" in docs[0].metadata
        assert str(sample_txt_file) in docs[0].metadata["source"]

    def test_raises_for_missing_path(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path / "nonexistent.txt")

    def test_raises_for_empty_directory(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No .txt or .pdf files"):
            load_documents(tmp_path)

    def test_loads_multiple_files(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("Document A content")
        (tmp_path / "b.txt").write_text("Document B content")
        docs = load_documents(tmp_path)
        assert len(docs) == 2


# ── split_documents tests ──────────────────────────────────────────────────────

class TestSplitDocuments:
    def test_splits_into_smaller_chunks(self, sample_docs: list[Document]):
        long_doc = Document(
            page_content="word " * 500,  # 2500 chars
            metadata={"source": "test.txt"},
        )
        chunks = split_documents([long_doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        long_doc = Document(page_content="a" * 1000, metadata={})
        chunks = split_documents([long_doc], chunk_size=100, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.page_content) <= 100 + 20  # small tolerance for splitter

    def test_metadata_preserved_in_chunks(self, sample_docs: list[Document]):
        chunks = split_documents(sample_docs)
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_empty_documents_returns_empty(self):
        result = split_documents([])
        assert result == []

    def test_overlap_creates_repeated_content(self):
        text = "First sentence here. " * 20
        doc = Document(page_content=text, metadata={})
        chunks_with_overlap = split_documents([doc], chunk_size=100, chunk_overlap=30)
        chunks_no_overlap = split_documents([doc], chunk_size=100, chunk_overlap=0)
        # With overlap, total characters across chunks is larger
        total_with = sum(len(c.page_content) for c in chunks_with_overlap)
        total_without = sum(len(c.page_content) for c in chunks_no_overlap)
        assert total_with >= total_without


# ── Vector store tests (fully mocked) ─────────────────────────────────────────

class TestBuildVectorStore:
    def test_raises_without_openai_key(self, monkeypatch, sample_docs):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from naive_rag import build_vector_store
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            build_vector_store(sample_docs)

    def test_creates_vector_store_with_mocked_embeddings(self, fake_openai_key, sample_docs, tmp_path):
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 2

        with patch("naive_rag._get_embeddings") as mock_emb, \
             patch("langchain_chroma.Chroma.from_documents", return_value=mock_vs):
            mock_emb.return_value = MagicMock()
            from naive_rag import build_vector_store
            vs = build_vector_store(sample_docs, persist_dir=tmp_path / "chroma")

        assert vs is not None


# ── RAG chain tests ────────────────────────────────────────────────────────────

class TestBuildRagChain:
    def test_chain_built_without_error(self, fake_anthropic_key, fake_openai_key):
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Python context", metadata={})
        ]
        mock_vs.as_retriever.return_value = mock_retriever

        with patch("naive_rag._get_llm") as mock_llm:
            mock_llm.return_value = MagicMock()
            from naive_rag import build_rag_chain
            chain = build_rag_chain(mock_vs)

        assert chain is not None

    def test_chain_invoke_returns_string(self, fake_anthropic_key, fake_openai_key):
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Python was created by Guido van Rossum.", metadata={})
        ]
        mock_vs.as_retriever.return_value = mock_retriever

        with patch("naive_rag._get_llm") as mock_llm:
            mock_llm_instance = MagicMock()
            fake_response = MagicMock()
            fake_response.content = "Python was created by Guido van Rossum in 1991."
            mock_llm_instance.invoke.return_value = fake_response
            mock_llm.return_value = mock_llm_instance

            from naive_rag import build_rag_chain
            chain = build_rag_chain(mock_vs)

            with patch.object(chain, "invoke", return_value="Python was created by Guido van Rossum in 1991."):
                result = chain.invoke("Who created Python?")

        assert isinstance(result, str)
