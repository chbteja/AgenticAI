"""
Root-level pytest fixtures shared across all phases.

Import in any test with: from conftest import ...  (pytest auto-discovers this)
"""

import os
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clear_sensitive_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never accidentally use real API keys from the host environment."""
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Inject a fake Anthropic API key so client construction doesn't fail."""
    key = "sk-ant-test-key-0000000000000000"
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    return key


@pytest.fixture
def fake_openai_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Inject a fake OpenAI API key so client construction doesn't fail."""
    key = "sk-test-key-0000000000000000"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    return key


@pytest.fixture
def fake_cohere_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = "test-cohere-key-0000"
    monkeypatch.setenv("COHERE_API_KEY", key)
    return key


@pytest.fixture
def sample_documents() -> list[dict]:
    """Small document set used by RAG tests."""
    return [
        {"text": "The Eiffel Tower is located in Paris, France.", "source": "geography"},
        {"text": "Python was created by Guido van Rossum in 1991.", "source": "tech"},
        {"text": "Photosynthesis converts sunlight into glucose.", "source": "biology"},
        {"text": "The speed of light is approximately 299,792,458 metres per second.", "source": "physics"},
        {"text": "Water boils at 100 degrees Celsius at sea level.", "source": "chemistry"},
    ]


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Temporary directory cleaned up after each test — safe for vector store data."""
    with tempfile.TemporaryDirectory() as d:
        yield d
