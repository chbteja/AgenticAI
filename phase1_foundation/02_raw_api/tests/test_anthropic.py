"""Tests for anthropic_basics.py — all API calls mocked."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCreateClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from anthropic_basics import create_client
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            create_client()

    def test_creates_client_with_key(self, fake_anthropic_key):
        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            from anthropic_basics import create_client
            client = create_client()
        assert client is not None


class TestSimpleCompletion:
    def _mock_response(self, text: str, input_tokens: int = 10, output_tokens: int = 20):
        mock = MagicMock()
        mock.content = [MagicMock(text=text)]
        mock.usage.input_tokens = input_tokens
        mock.usage.output_tokens = output_tokens
        return mock

    def test_returns_string(self, fake_anthropic_key):
        from anthropic_basics import simple_completion
        with patch("anthropic_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = self._mock_response("Hello!")
            mock_factory.return_value = mock_client

            result = simple_completion("Say hello")
        assert isinstance(result, str)
        assert result == "Hello!"

    def test_empty_prompt_still_calls_api(self, fake_anthropic_key):
        from anthropic_basics import simple_completion
        with patch("anthropic_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = self._mock_response("OK")
            mock_factory.return_value = mock_client

            result = simple_completion("")
        assert result == "OK"

    def test_system_prompt_included_in_kwargs(self, fake_anthropic_key):
        from anthropic_basics import simple_completion
        with patch("anthropic_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = self._mock_response("Arr!")
            mock_factory.return_value = mock_client

            simple_completion("Hello", system="You are a pirate.")
            call_kwargs = mock_client.messages.create.call_args[1]

        assert "system" in call_kwargs
        assert call_kwargs["system"] == "You are a pirate."

    def test_re_raises_api_error(self, fake_anthropic_key):
        from anthropic import APIError
        from anthropic_basics import simple_completion

        with patch("anthropic_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = APIError(
                message="Server error", request=MagicMock(), body={}
            )
            mock_factory.return_value = mock_client

            with pytest.raises(APIError):
                simple_completion("test")


class TestStreamCompletion:
    def test_yields_string_chunks(self, fake_anthropic_key):
        from anthropic_basics import stream_completion

        chunks = ["Hello", " ", "world", "!"]

        with patch("anthropic_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=False)
            mock_stream.text_stream = iter(chunks)
            mock_client.messages.stream.return_value = mock_stream
            mock_factory.return_value = mock_client

            result = list(stream_completion("Count to 3"))

        assert result == chunks

    def test_empty_response_yields_nothing(self, fake_anthropic_key):
        from anthropic_basics import stream_completion

        with patch("anthropic_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=False)
            mock_stream.text_stream = iter([])
            mock_client.messages.stream.return_value = mock_stream
            mock_factory.return_value = mock_client

            result = list(stream_completion("Say nothing"))

        assert result == []


class TestMultiTurnConversation:
    def _mock_response(self, text: str):
        mock = MagicMock()
        mock.content = [MagicMock(text=text)]
        mock.usage.input_tokens = 10
        mock.usage.output_tokens = 5
        return mock

    def test_raises_on_empty_messages(self, fake_anthropic_key):
        from anthropic_basics import multi_turn_conversation
        with pytest.raises(ValueError, match="empty"):
            multi_turn_conversation([])

    def test_raises_if_first_message_not_user(self, fake_anthropic_key):
        from anthropic_basics import multi_turn_conversation
        with pytest.raises(ValueError, match="user message"):
            multi_turn_conversation([{"role": "assistant", "content": "hi"}])

    def test_history_grows_by_one_assistant_turn(self, fake_anthropic_key):
        from anthropic_basics import multi_turn_conversation

        initial = [{"role": "user", "content": "Hello"}]
        with patch("anthropic_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = self._mock_response("Hi there!")
            mock_factory.return_value = mock_client

            reply, updated = multi_turn_conversation(initial)

        assert reply == "Hi there!"
        assert len(updated) == len(initial) + 1
        assert updated[-1]["role"] == "assistant"
        assert updated[-1]["content"] == "Hi there!"

    def test_full_history_sent_to_api(self, fake_anthropic_key):
        from anthropic_basics import multi_turn_conversation

        messages = [
            {"role": "user", "content": "My name is Bob."},
            {"role": "assistant", "content": "Hello Bob!"},
            {"role": "user", "content": "What is my name?"},
        ]
        with patch("anthropic_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = self._mock_response("Your name is Bob.")
            mock_factory.return_value = mock_client

            multi_turn_conversation(messages)
            call_kwargs = mock_client.messages.create.call_args[1]

        assert call_kwargs["messages"] == messages
