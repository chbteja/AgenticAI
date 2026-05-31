"""Tests for langsmith_demo.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langsmith_demo import _is_tracing_enabled, log_feedback_to_run, setup_tracing


class TestIsTracingEnabled:
    def test_false_without_api_key(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        assert _is_tracing_enabled() is False

    def test_false_with_key_but_tracing_not_set(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        assert _is_tracing_enabled() is False

    def test_false_with_tracing_false(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
        assert _is_tracing_enabled() is False

    def test_true_with_both_set(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        assert _is_tracing_enabled() is True

    def test_case_insensitive_tracing_value(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "TRUE")
        assert _is_tracing_enabled() is True


class TestSetupTracing:
    def test_sets_environment_variables(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
        setup_tracing("test-project")
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"

    def test_no_error_without_api_key(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        # Should not raise — just logs a warning
        setup_tracing("test-project")

    def test_default_project_name(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
        setup_tracing()
        assert "agentic-ai-curriculum" in os.environ.get("LANGCHAIN_PROJECT", "")


class TestLogFeedbackToRun:
    def test_does_nothing_without_tracing(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        # Should not raise — just logs a warning
        log_feedback_to_run("run-id-123", 0.8)

    def test_calls_langsmith_client_when_tracing_enabled(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

        mock_client = MagicMock()
        with patch("langsmith.Client", return_value=mock_client):
            log_feedback_to_run("run-id-123", 0.9, "faithfulness", "Good answer")

        mock_client.create_feedback.assert_called_once_with(
            run_id="run-id-123",
            key="faithfulness",
            score=0.9,
            comment="Good answer",
        )

    def test_handles_langsmith_api_error_gracefully(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__test")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

        mock_client = MagicMock()
        mock_client.create_feedback.side_effect = Exception("API error")

        with patch("langsmith.Client", return_value=mock_client):
            # Should not raise — logs the error instead
            log_feedback_to_run("run-id-123", 0.5)


class TestRunWithTracing:
    def test_returns_expected_keys(self, fake_anthropic_key):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "An answer."

        from langsmith_demo import run_with_tracing
        result = run_with_tracing("What is Python?", mock_chain)

        assert "question" in result
        assert "answer" in result
        assert "run_id" in result
        assert "latency_ms" in result

    def test_answer_matches_chain_output(self, fake_anthropic_key):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Python is great."

        from langsmith_demo import run_with_tracing
        result = run_with_tracing("What is Python?", mock_chain)

        assert result["answer"] == "Python is great."

    def test_latency_is_non_negative(self, fake_anthropic_key):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "ok"

        from langsmith_demo import run_with_tracing
        result = run_with_tracing("test", mock_chain)

        assert result["latency_ms"] >= 0

    def test_run_id_is_uuid_format(self, fake_anthropic_key):
        import uuid
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "ok"

        from langsmith_demo import run_with_tracing
        result = run_with_tracing("test", mock_chain)

        # Should be a valid UUID string
        uuid.UUID(result["run_id"])  # Raises ValueError if invalid
