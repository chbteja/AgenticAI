"""Tests for langgraph_agent.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph_agent import calculate, lookup_fact, should_continue, AgentState


class TestCalculateTool:
    def test_basic_math(self):
        result = calculate.invoke({"expression": "10 + 5"})
        assert "15" in result

    def test_power(self):
        result = calculate.invoke({"expression": "2 ** 8"})
        assert "256" in result

    def test_rejects_unsafe(self):
        result = calculate.invoke({"expression": "os.getcwd()"})
        assert "Error" in result

    def test_division(self):
        result = calculate.invoke({"expression": "100 / 4"})
        assert "25" in result


class TestLookupFactTool:
    def test_python_fact(self):
        result = lookup_fact.invoke({"topic": "python"})
        assert "Guido" in result or "1991" in result

    def test_case_insensitive(self):
        lower = lookup_fact.invoke({"topic": "rag"})
        upper = lookup_fact.invoke({"topic": "RAG"})
        assert lower == upper

    def test_unknown_topic_returns_available_list(self):
        result = lookup_fact.invoke({"topic": "blahblah"})
        assert "No fact found" in result
        assert "python" in result.lower()

    def test_langgraph_topic(self):
        result = lookup_fact.invoke({"topic": "langgraph"})
        assert "LangGraph" in result


class TestShouldContinue:
    def _make_state(self, has_tool_calls: bool, step_count: int = 0) -> AgentState:
        if has_tool_calls:
            ai_msg = MagicMock(spec=AIMessage)
            ai_msg.tool_calls = [{"name": "calculate", "args": {"expression": "1+1"}, "id": "123"}]
        else:
            ai_msg = AIMessage(content="I have the answer.")
        return {"messages": [ai_msg], "step_count": step_count}

    def test_routes_to_tools_when_tool_calls_present(self):
        state = self._make_state(has_tool_calls=True)
        result = should_continue(state)
        assert result == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        state = self._make_state(has_tool_calls=False)
        result = should_continue(state)
        assert result == "end"

    def test_routes_to_end_at_max_steps(self):
        state = self._make_state(has_tool_calls=True, step_count=10)
        result = should_continue(state)
        assert result == "end"

    def test_does_not_end_at_step_9(self):
        state = self._make_state(has_tool_calls=True, step_count=9)
        result = should_continue(state)
        assert result == "tools"


class TestBuildGraph:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from langgraph_agent import build_graph
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            build_graph()

    def test_returns_compiled_graph(self, fake_anthropic_key):
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_cls.return_value = mock_llm

            from langgraph_agent import build_graph
            graph = build_graph()

        assert graph is not None


class TestRunGraph:
    def test_returns_expected_dict_keys(self, fake_anthropic_key):
        mock_graph = MagicMock()
        ai_response = AIMessage(content="The answer is 42.")
        mock_graph.invoke.return_value = {
            "messages": [HumanMessage(content="test"), ai_response],
            "step_count": 1,
        }

        from langgraph_agent import run_graph
        result = run_graph("What is 6 * 7?", mock_graph)

        assert "question" in result
        assert "answer" in result
        assert "steps" in result
        assert result["answer"] == "The answer is 42."

    def test_extracts_tool_steps(self, fake_anthropic_key):
        mock_graph = MagicMock()
        ai_with_tools = MagicMock(spec=AIMessage)
        ai_with_tools.tool_calls = [{"name": "calculate", "args": {"expression": "2+2"}, "id": "1"}]
        final_ai = AIMessage(content="The result is 4.")

        mock_graph.invoke.return_value = {
            "messages": [HumanMessage(content="test"), ai_with_tools, final_ai],
            "step_count": 2,
        }

        from langgraph_agent import run_graph
        result = run_graph("What is 2+2?", mock_graph)

        assert len(result["steps"]) == 1
        assert result["steps"][0]["tool"] == "calculate"
