"""Tests for multi_query_rag.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_doc(content: str) -> Document:
    return Document(page_content=content, metadata={"source": "test"})


class TestGenerateQueryVariants:
    def test_returns_list_of_strings(self, fake_anthropic_key):
        variants_text = "What Python libraries build web apps?\nHow to make websites with Python?\nPython backend frameworks?"
        with patch("langchain_core.runnables.base.RunnableSequence.invoke", return_value=variants_text):
            from multi_query_rag import generate_query_variants
            result = generate_query_variants("Python web frameworks", n=3)
        assert isinstance(result, list)
        assert all(isinstance(v, str) for v in result)

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from multi_query_rag import generate_query_variants
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            generate_query_variants("test")

    def test_original_query_not_in_variants(self, fake_anthropic_key):
        query = "Python web frameworks"
        variants_text = f"{query}\nAlternative version one\nAlternative version two"
        with patch("langchain_core.runnables.base.RunnableSequence.invoke", return_value=variants_text):
            from multi_query_rag import generate_query_variants
            result = generate_query_variants(query, n=3)
        assert query not in result


class TestMultiQueryRetrieve:
    def _make_retriever(self, docs_by_query: dict):
        """Create a mock retriever that returns different docs for different queries."""
        mock = MagicMock()
        def invoke_side_effect(query):
            return docs_by_query.get(query, [])
        mock.invoke.side_effect = invoke_side_effect
        return mock

    def test_returns_list_of_documents(self, fake_anthropic_key, fake_openai_key):
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock(
            invoke=MagicMock(return_value=[_make_doc("content")])
        )
        with patch("multi_query_rag.generate_query_variants", return_value=["variant1", "variant2"]):
            from multi_query_rag import multi_query_retrieve
            result = multi_query_retrieve("query", mock_vs, n_variants=2, k_per_query=2)
        assert isinstance(result, list)

    def test_deduplicates_identical_docs(self, fake_anthropic_key, fake_openai_key):
        shared_doc = _make_doc("This content appears in both query results.")
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock(
            invoke=MagicMock(return_value=[shared_doc])
        )
        with patch("multi_query_rag.generate_query_variants", return_value=["variant1"]):
            from multi_query_rag import multi_query_retrieve
            result = multi_query_retrieve("query", mock_vs, n_variants=1, k_per_query=3)
        # Both queries returned the same doc — should appear only once
        assert len(result) == 1

    def test_union_of_unique_docs(self, fake_anthropic_key, fake_openai_key):
        doc_a = _make_doc("Document A with unique content about Python.")
        doc_b = _make_doc("Document B with unique content about frameworks.")
        doc_c = _make_doc("Document C with unique content about web development.")

        call_count = [0]
        def retriever_invoke(_):
            call_count[0] += 1
            return [doc_a] if call_count[0] == 1 else [doc_b, doc_c]

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock(invoke=retriever_invoke)

        with patch("multi_query_rag.generate_query_variants", return_value=["variant1"]):
            from multi_query_rag import multi_query_retrieve
            result = multi_query_retrieve("query", mock_vs, n_variants=1)

        # Should have all 3 unique documents
        assert len(result) == 3

    def test_includes_original_query_in_retrieval(self, fake_anthropic_key, fake_openai_key):
        queried_with = []

        def track_invoke(q):
            queried_with.append(q)
            return []

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock(invoke=track_invoke)

        with patch("multi_query_rag.generate_query_variants", return_value=["variant1"]):
            from multi_query_rag import multi_query_retrieve
            multi_query_retrieve("original query", mock_vs, n_variants=1)

        assert "original query" in queried_with
