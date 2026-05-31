"""Tests for react_agent.py — all LLM calls mocked."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from react_agent import calculator, knowledge_search


class TestCalculator:
    def test_basic_addition(self):
        result = calculator.invoke({"expression": "2 + 2"})
        assert "4" in result

    def test_multiplication(self):
        result = calculator.invoke({"expression": "6 * 7"})
        assert "42" in result

    def test_complex_expression(self):
        result = calculator.invoke({"expression": "(10 + 5) * 2"})
        assert "30" in result

    def test_power(self):
        result = calculator.invoke({"expression": "2 ** 10"})
        assert "1024" in result

    def test_float_division(self):
        result = calculator.invoke({"expression": "10 / 4"})
        assert "2.5" in result

    def test_rejects_dangerous_input(self):
        result = calculator.invoke({"expression": "__import__('os').system('ls')"})
        assert "Error" in result

    def test_rejects_letters(self):
        result = calculator.invoke({"expression": "os.getcwd()"})
        assert "Error" in result

    def test_empty_expression_handled(self):
        # Should not crash, returns some kind of result or error
        result = calculator.invoke({"expression": ""})
        assert isinstance(result, str)

    def test_returns_string(self):
        result = calculator.invoke({"expression": "100"})
        assert isinstance(result, str)


class TestKnowledgeSearch:
    def test_finds_python(self):
        result = knowledge_search.invoke({"query": "python"})
        assert "Python" in result
        assert "Guido" in result

    def test_finds_rag(self):
        result = knowledge_search.invoke({"query": "rag"})
        assert "RAG" in result or "Retrieval" in result

    def test_finds_langchain(self):
        result = knowledge_search.invoke({"query": "langchain"})
        assert "LangChain" in result

    def test_partial_match_works(self):
        result = knowledge_search.invoke({"query": "vector"})
        # Should match "vector database"
        assert "vector" in result.lower()

    def test_unknown_topic_returns_helpful_message(self):
        result = knowledge_search.invoke({"query": "quantum gravity wormhole"})
        assert "No information found" in result
        assert "Available topics" in result

    def test_returns_string(self):
        result = knowledge_search.invoke({"query": "python"})
        assert isinstance(result, str)

    def test_case_insensitive(self):
        lower = knowledge_search.invoke({"query": "python"})
        upper = knowledge_search.invoke({"query": "PYTHON"})
        assert lower == upper  # Both should find the entry


class TestSummariseText:
    def test_short_text_returns_too_short_message(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from react_agent import summarise_text
        result = summarise_text.invoke({"text": "Too short"})
        assert "too short" in result.lower() or "short" in result.lower()

    def test_no_api_key_returns_error_message(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from react_agent import summarise_text
        long_text = "This is a long enough text to summarise. " * 10
        result = summarise_text.invoke({"text": long_text})
        assert "ANTHROPIC_API_KEY" in result or "Cannot" in result

    def test_calls_api_and_returns_summary(self, fake_anthropic_key):
        long_text = "Python is amazing. " * 50
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="• Python is used widely\n• It is powerful")]

        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client

            from react_agent import summarise_text
            result = summarise_text.invoke({"text": long_text})

        assert isinstance(result, str)
        assert len(result) > 0


class TestBuildReactAgent:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from react_agent import build_react_agent
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            build_react_agent()

    def test_returns_executor(self, fake_anthropic_key):
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            with patch("langchain.agents.create_tool_calling_agent") as mock_create, \
                 patch("langchain.agents.AgentExecutor") as mock_executor_cls:
                mock_create.return_value = MagicMock()
                mock_executor_cls.return_value = MagicMock()

                from react_agent import build_react_agent
                agent = build_react_agent()

        assert agent is not None

    def test_agent_has_three_tools(self, fake_anthropic_key):
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            with patch("langchain.agents.create_tool_calling_agent") as mock_create, \
                 patch("langchain.agents.AgentExecutor") as mock_executor_cls:
                mock_create.return_value = MagicMock()
                mock_executor_cls.return_value = MagicMock()

                from react_agent import build_react_agent
                build_react_agent()

                # Check that 3 tools were passed to create_tool_calling_agent
                tools_arg = mock_create.call_args[0][1]
        assert len(tools_arg) == 3
