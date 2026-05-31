"""Tests for hyde_rag.py — all LLM and vector store calls mocked."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGenerateHypotheticalAnswer:
    def test_returns_non_empty_string(self, fake_anthropic_key):
        with patch("hyde_rag._get_llm") as mock_llm:
            mock_chain = MagicMock()
            mock_chain.__or__ = lambda self, other: mock_chain
            mock_llm_inst = MagicMock()
            mock_llm.return_value = mock_llm_inst

            from hyde_rag import generate_hypothetical_answer
            with patch("hyde_rag.StrOutputParser") as mock_parser:
                # Mock the entire chain pipeline
                with patch.object(
                    type(mock_llm_inst).__or__,
                    "__get__",
                    return_value=lambda self, x: MagicMock(invoke=lambda d: "A hypothetical answer about Python."),
                    create=True,
                ):
                    pass

            # Simpler: mock at the chain.invoke level
            with patch("langchain_core.runnables.base.RunnableSequence.invoke",
                       return_value="Python was created by Guido van Rossum in 1991."):
                result = generate_hypothetical_answer("Who created Python?")

        assert isinstance(result, str)

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from hyde_rag import generate_hypothetical_answer
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            generate_hypothetical_answer("test")


class TestNaiveSearch:
    def test_returns_list_of_documents(self, fake_openai_key):
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Python content", metadata={}),
        ]
        mock_vs.as_retriever.return_value = mock_retriever

        from hyde_rag import naive_search
        docs = naive_search("Python", mock_vs, k=2)

        assert isinstance(docs, list)
        assert len(docs) == 1
        mock_retriever.invoke.assert_called_once_with("Python")

    def test_k_passed_to_retriever(self, fake_openai_key):
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[]))

        from hyde_rag import naive_search
        naive_search("test", mock_vs, k=7)

        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 7})


class TestHydeSearch:
    def test_uses_hypothetical_answer_for_search(self, fake_openai_key, fake_anthropic_key):
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        returned_doc = Document(page_content="Python was created by Guido.", metadata={})
        mock_retriever.invoke.return_value = [returned_doc]
        mock_vs.as_retriever.return_value = mock_retriever

        hypothetical = "Python was created by Guido van Rossum in 1991 in the Netherlands."
        with patch("hyde_rag.generate_hypothetical_answer", return_value=hypothetical):
            from hyde_rag import hyde_search
            docs = hyde_search("Python creator", mock_vs, k=3)

        # The retriever should have been called with the hypothetical answer, not the query
        mock_retriever.invoke.assert_called_once_with(hypothetical)
        assert len(docs) == 1

    def test_returns_list(self, fake_openai_key, fake_anthropic_key):
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[]))

        with patch("hyde_rag.generate_hypothetical_answer", return_value="A hypothetical answer."):
            from hyde_rag import hyde_search
            result = hyde_search("query", mock_vs)

        assert isinstance(result, list)


class TestCompareNaiveVsHyde:
    def test_returns_comparison_dict(self, fake_openai_key, fake_anthropic_key):
        mock_vs = MagicMock()
        doc = Document(page_content="Some content about Python", metadata={})
        mock_retriever = MagicMock(invoke=MagicMock(return_value=[doc]))
        mock_vs.as_retriever.return_value = mock_retriever

        with patch("hyde_rag.generate_hypothetical_answer", return_value="Hypothetical answer."):
            from hyde_rag import compare_naive_vs_hyde
            result = compare_naive_vs_hyde("Python creator", mock_vs)

        assert "query" in result
        assert "naive_first_chunk" in result
        assert "hyde_first_chunk" in result
        assert result["query"] == "Python creator"

    def test_handles_empty_retrieval(self, fake_openai_key, fake_anthropic_key):
        mock_vs = MagicMock()
        mock_retriever = MagicMock(invoke=MagicMock(return_value=[]))
        mock_vs.as_retriever.return_value = mock_retriever

        with patch("hyde_rag.generate_hypothetical_answer", return_value="Hypothetical."):
            from hyde_rag import compare_naive_vs_hyde
            result = compare_naive_vs_hyde("empty query", mock_vs)

        assert result["naive_first_chunk"] is None
        assert result["hyde_first_chunk"] is None
