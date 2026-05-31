"""
Tests for tokens_demo.py

All tests run offline — no API keys required.
API-dependent functions (count_tokens_anthropic) are mocked.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokens_demo import (
    count_tokens_tiktoken,
    estimate_cost,
    inspect_token_breakdown,
)


class TestCountTokensTiktoken:
    def test_basic_english_sentence(self):
        count = count_tokens_tiktoken("Hello, world!")
        # "Hello" "," " world" "!" = approximately 4 tokens
        assert isinstance(count, int)
        assert count > 0

    def test_empty_string_returns_zero(self):
        assert count_tokens_tiktoken("") == 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens_tiktoken("Hi")
        long = count_tokens_tiktoken("Hi " * 100)
        assert long > short

    def test_unknown_model_falls_back_gracefully(self):
        # Should not raise, should fall back to cl100k_base
        count = count_tokens_tiktoken("Hello", model="unknown-model-xyz")
        assert isinstance(count, int)
        assert count > 0

    def test_code_tokenizes(self):
        code = "def add(a, b):\n    return a + b"
        count = count_tokens_tiktoken(code)
        assert count > 0

    def test_returns_integer(self):
        result = count_tokens_tiktoken("test string")
        assert isinstance(result, int)


class TestInspectTokenBreakdown:
    def test_returns_list_of_tuples(self):
        result = inspect_token_breakdown("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_token_ids_are_integers(self):
        result = inspect_token_breakdown("Hello")
        for tid, _ in result:
            assert isinstance(tid, int)

    def test_decoded_pieces_are_strings(self):
        result = inspect_token_breakdown("Hello")
        for _, piece in result:
            assert isinstance(piece, str)

    def test_empty_string_returns_empty_list(self):
        assert inspect_token_breakdown("") == []

    def test_reassembled_text_matches_original(self):
        text = "Tokenization is interesting."
        breakdown = inspect_token_breakdown(text)
        reassembled = "".join(piece for _, piece in breakdown)
        assert reassembled == text


class TestEstimateCost:
    def test_returns_dict_with_required_keys(self):
        result = estimate_cost(1000, 500, "gpt-4o")
        assert "input_cost" in result
        assert "output_cost" in result
        assert "total_cost" in result

    def test_total_equals_sum_of_parts(self):
        result = estimate_cost(10_000, 2_000, "gpt-4o-mini")
        assert abs(result["total_cost"] - (result["input_cost"] + result["output_cost"])) < 1e-9

    def test_more_tokens_cost_more(self):
        low = estimate_cost(1_000, 100, "claude-sonnet-4-6")
        high = estimate_cost(100_000, 10_000, "claude-sonnet-4-6")
        assert high["total_cost"] > low["total_cost"]

    def test_zero_tokens_zero_cost(self):
        result = estimate_cost(0, 0, "gpt-4o")
        assert result["total_cost"] == 0.0

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model"):
            estimate_cost(1000, 500, "nonexistent-model-xyz")

    def test_all_known_models_compute_without_error(self):
        from tokens_demo import MODEL_PRICING
        for model in MODEL_PRICING:
            result = estimate_cost(1000, 200, model)
            assert result["total_cost"] >= 0


class TestCountTokensAnthropic:
    def test_returns_none_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from tokens_demo import count_tokens_anthropic
        result = count_tokens_anthropic("Hello world")
        assert result is None

    def test_returns_integer_when_api_responds(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        mock_response = MagicMock()
        mock_response.input_tokens = 42

        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_client.messages.count_tokens.return_value = mock_response
            mock_anthropic_cls.return_value = mock_client

            from tokens_demo import count_tokens_anthropic
            result = count_tokens_anthropic("Hello world")

        assert result == 42

    def test_returns_none_on_api_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_client.messages.count_tokens.side_effect = Exception("API error")
            mock_anthropic_cls.return_value = mock_client

            from tokens_demo import count_tokens_anthropic
            result = count_tokens_anthropic("Hello world")

        assert result is None
