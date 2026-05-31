"""Tests for reranking_rag.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_doc(content: str) -> Document:
    return Document(page_content=content, metadata={"source": "test"})


class TestRerankWithFlashrank:
    def test_returns_list_of_documents(self, monkeypatch):
        docs = [_make_doc(f"Document {i} content about Python.") for i in range(5)]
        top2 = docs[:2]

        with patch("langchain_community.document_compressors.FlashrankRerank") as mock_cls:
            mock_compressor = MagicMock()
            mock_compressor.compress_documents.return_value = top2
            mock_cls.return_value = mock_compressor

            from reranking_rag import rerank_with_flashrank
            result = rerank_with_flashrank("Python web frameworks", docs, top_n=2)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_returns_truncated_docs_when_flashrank_unavailable(self):
        docs = [_make_doc(f"Document {i}") for i in range(10)]

        with patch.dict("sys.modules", {"flashrank": None, "langchain_community.document_compressors": MagicMock(
            **{"FlashrankRerank": MagicMock(side_effect=ImportError)}
        )}):
            from reranking_rag import rerank_with_flashrank
            with patch("langchain_community.document_compressors.FlashrankRerank", side_effect=ImportError):
                result = rerank_with_flashrank("query", docs, top_n=3)

        assert len(result) <= 3

    def test_top_n_respected(self, monkeypatch):
        docs = [_make_doc(f"Doc {i}") for i in range(10)]

        with patch("langchain_community.document_compressors.FlashrankRerank") as mock_cls:
            mock_compressor = MagicMock()
            mock_compressor.compress_documents.return_value = docs[:5]
            mock_cls.return_value = mock_compressor

            from reranking_rag import rerank_with_flashrank
            rerank_with_flashrank("query", docs, top_n=5)

            mock_cls.assert_called_with(top_n=5)


class TestRerankWithCohere:
    def test_raises_without_cohere_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        docs = [_make_doc("Some content")]
        from reranking_rag import rerank_with_cohere
        with pytest.raises(EnvironmentError, match="COHERE_API_KEY"):
            rerank_with_cohere("query", docs)

    def test_calls_cohere_api_with_key(self, fake_cohere_key):
        docs = [_make_doc("content")]

        with patch("langchain_community.document_compressors.CohereRerank") as mock_cls:
            mock_compressor = MagicMock()
            mock_compressor.compress_documents.return_value = docs
            mock_cls.return_value = mock_compressor

            from reranking_rag import rerank_with_cohere
            result = rerank_with_cohere("query", docs, top_n=1)

        assert result == docs


class TestRetrieveThenRerank:
    def test_returns_expected_dict_keys(self, fake_openai_key):
        docs = [_make_doc(f"Document {i} about Python frameworks.") for i in range(5)]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=docs))

        with patch("reranking_rag.rerank_with_flashrank", return_value=docs[:2]):
            from reranking_rag import retrieve_then_rerank
            result = retrieve_then_rerank("Python web", mock_vs, initial_k=5, final_k=2)

        assert "query" in result
        assert "initial_docs" in result
        assert "reranked_docs" in result
        assert "initial_top_snippet" in result
        assert "reranked_top_snippet" in result

    def test_initial_k_greater_than_final_k(self, fake_openai_key):
        """Confirm over-retrieve pattern: initial_k > final_k."""
        docs = [_make_doc(f"Doc {i}") for i in range(8)]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=docs))

        with patch("reranking_rag.rerank_with_flashrank", return_value=docs[:3]) as mock_rerank:
            from reranking_rag import retrieve_then_rerank
            result = retrieve_then_rerank("query", mock_vs, initial_k=8, final_k=3)

        assert len(result["initial_docs"]) == 8
        assert len(result["reranked_docs"]) == 3

    def test_handles_empty_retrieval(self, fake_openai_key):
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[]))

        with patch("reranking_rag.rerank_with_flashrank", return_value=[]):
            from reranking_rag import retrieve_then_rerank
            result = retrieve_then_rerank("query", mock_vs)

        assert result["initial_top_snippet"] is None
        assert result["reranked_top_snippet"] is None
