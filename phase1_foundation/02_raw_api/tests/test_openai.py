"""Tests for openai_basics.py — all API calls mocked."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _mock_completion(text: str, prompt_tokens: int = 10, completion_tokens: int = 20):
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=text))]
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    return mock


class TestCreateClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from openai_basics import create_client
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            create_client()

    def test_succeeds_with_key(self, fake_openai_key):
        with patch("openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            from openai_basics import create_client
            client = create_client()
        assert client is not None


class TestChatCompletion:
    def test_returns_string(self, fake_openai_key):
        from openai_basics import chat_completion
        with patch("openai_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_completion("Hello!")
            mock_factory.return_value = mock_client

            result = chat_completion([{"role": "user", "content": "Hi"}])
        assert result == "Hello!"

    def test_empty_messages_raises(self, fake_openai_key):
        from openai_basics import chat_completion
        with pytest.raises(ValueError, match="empty"):
            chat_completion([])

    def test_temperature_passed_to_api(self, fake_openai_key):
        from openai_basics import chat_completion
        with patch("openai_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_completion("ok")
            mock_factory.return_value = mock_client

            chat_completion([{"role": "user", "content": "test"}], temperature=0.0)
            call_kwargs = mock_client.chat.completions.create.call_args[1]

        assert call_kwargs["temperature"] == 0.0


class TestStreamChat:
    def test_yields_non_empty_deltas(self, fake_openai_key):
        from openai_basics import stream_chat

        deltas = ["Hello", " world", "!"]
        mock_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content=d))])
            for d in deltas
        ]

        with patch("openai_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter(mock_chunks)
            mock_factory.return_value = mock_client

            result = list(stream_chat([{"role": "user", "content": "test"}]))

        assert result == deltas

    def test_none_deltas_are_skipped(self, fake_openai_key):
        from openai_basics import stream_chat

        # OpenAI sometimes sends None deltas at the end of the stream
        mock_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="text"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
        ]

        with patch("openai_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter(mock_chunks)
            mock_factory.return_value = mock_client

            result = list(stream_chat([{"role": "user", "content": "test"}]))

        assert None not in result
        assert "text" in result


class TestMultiTurnConversation:
    def test_user_message_appended_to_history(self, fake_openai_key):
        from openai_basics import multi_turn_conversation
        history: list[dict] = []

        with patch("openai_basics.chat_completion", return_value="I'm fine!"):
            reply, new_history = multi_turn_conversation("How are you?", history)

        assert reply == "I'm fine!"
        assert any(m["role"] == "user" and "How are you?" in m["content"] for m in new_history)
        assert any(m["role"] == "assistant" and "I'm fine!" in m["content"] for m in new_history)

    def test_system_prompt_added_on_first_turn(self, fake_openai_key):
        from openai_basics import multi_turn_conversation

        with patch("openai_basics.chat_completion", return_value="Aye!") as mock_cc:
            multi_turn_conversation("Hello", [], system="You are a pirate.")
            messages_arg = mock_cc.call_args[0][0]

        assert messages_arg[0]["role"] == "system"
        assert "pirate" in messages_arg[0]["content"]

    def test_system_prompt_not_added_on_subsequent_turns(self, fake_openai_key):
        from openai_basics import multi_turn_conversation

        existing_history = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Ahoy"},
            {"role": "assistant", "content": "Arr!"},
        ]
        with patch("openai_basics.chat_completion", return_value="Aye!") as mock_cc:
            multi_turn_conversation("What's next?", existing_history, system="You are a pirate.")
            messages_arg = mock_cc.call_args[0][0]

        system_messages = [m for m in messages_arg if m["role"] == "system"]
        assert len(system_messages) == 1  # Not duplicated


class TestStructuredJsonOutput:
    def test_returns_parsed_dict(self, fake_openai_key):
        from openai_basics import structured_json_output

        fake_json = '{"name": "Alice", "age": 30}'
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=fake_json))]

        with patch("openai_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_factory.return_value = mock_client

            result = structured_json_output("Extract info", '{"name": "str", "age": "int"}')

        assert isinstance(result, dict)
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_invalid_json_raises_decode_error(self, fake_openai_key):
        from openai_basics import structured_json_output

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="not valid json {{{"))]

        with patch("openai_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_factory.return_value = mock_client

            with pytest.raises(json.JSONDecodeError):
                structured_json_output("test", "{}")

    def test_response_format_set_to_json_object(self, fake_openai_key):
        from openai_basics import structured_json_output

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='{"key": "value"}'))]

        with patch("openai_basics.create_client") as mock_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_factory.return_value = mock_client

            structured_json_output("test", "{}")
            call_kwargs = mock_client.chat.completions.create.call_args[1]

        assert call_kwargs["response_format"] == {"type": "json_object"}
