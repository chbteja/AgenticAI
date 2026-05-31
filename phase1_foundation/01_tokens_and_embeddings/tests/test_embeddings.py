"""
Tests for embeddings_demo.py

All external API calls are mocked — no OPENAI_API_KEY required to run tests.
"""

import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embeddings_demo import (
    batch_get_embeddings,
    cosine_similarity,
    get_embedding,
    semantic_search,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_fake_embedding(seed: float = 0.5, dim: int = 1536) -> list[float]:
    """Generate a deterministic fake embedding vector for testing."""
    import math
    return [math.sin(seed + i * 0.01) for i in range(dim)]


def _mock_openai_embedding(texts: list[str], dim: int = 4):
    """Return a mock OpenAI embeddings response for a list of texts."""
    mock_resp = MagicMock()
    mock_resp.data = [
        MagicMock(index=i, embedding=_make_fake_embedding(float(i), dim))
        for i in range(len(texts))
    ]
    return mock_resp


# ── cosine_similarity tests ───────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 2.0, 3.0]
        result = cosine_similarity(v, v)
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        result = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(result) < 1e-6

    def test_opposite_vectors_return_negative_one(self):
        result = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(result + 1.0) < 1e-6

    def test_result_bounded_between_minus_one_and_one(self):
        import random
        random.seed(42)
        v_a = [random.gauss(0, 1) for _ in range(64)]
        v_b = [random.gauss(0, 1) for _ in range(64)]
        result = cosine_similarity(v_a, v_b)
        assert -1.0 <= result <= 1.0

    def test_mismatched_lengths_raise_value_error(self):
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_raises_value_error(self):
        with pytest.raises(ValueError, match="zero vectors"):
            cosine_similarity([0.0, 0.0], [1.0, 2.0])

    def test_symmetry(self):
        v_a = [0.3, 0.7, 0.1]
        v_b = [0.6, 0.2, 0.9]
        assert abs(cosine_similarity(v_a, v_b) - cosine_similarity(v_b, v_a)) < 1e-9


# ── get_embedding tests ───────────────────────────────────────────────────────

class TestGetEmbedding:
    def test_empty_text_raises_value_error(self, fake_openai_key):
        with pytest.raises(ValueError, match="empty"):
            get_embedding("")

    def test_whitespace_only_raises_value_error(self, fake_openai_key):
        with pytest.raises(ValueError, match="empty"):
            get_embedding("   ")

    def test_missing_api_key_raises_environment_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            get_embedding("Hello world")

    def test_returns_list_of_floats(self, fake_openai_key):
        fake_vec = _make_fake_embedding(0.5, 1536)

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=fake_vec)]
            mock_client.embeddings.create.return_value = mock_resp
            mock_cls.return_value = mock_client

            result = get_embedding("Hello world")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
        assert len(result) == 1536


# ── batch_get_embeddings tests ────────────────────────────────────────────────

class TestBatchGetEmbeddings:
    def test_empty_list_raises_value_error(self, fake_openai_key):
        with pytest.raises(ValueError, match="empty"):
            batch_get_embeddings([])

    def test_list_with_empty_string_raises_value_error(self, fake_openai_key):
        with pytest.raises(ValueError):
            batch_get_embeddings(["hello", ""])

    def test_returns_one_vector_per_input(self, fake_openai_key):
        texts = ["first", "second", "third"]
        fake_resp = _mock_openai_embedding(texts, dim=8)

        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = fake_resp
            mock_cls.return_value = mock_client

            results = batch_get_embeddings(texts)

        assert len(results) == len(texts)
        assert all(isinstance(v, list) for v in results)


# ── semantic_search tests ─────────────────────────────────────────────────────

class TestSemanticSearch:
    def test_empty_documents_returns_empty_list(self, fake_openai_key):
        with patch("embeddings_demo.get_embedding", return_value=[0.5] * 4):
            result = semantic_search("query", [])
        assert result == []

    def test_returns_at_most_top_k_results(self, fake_openai_key):
        docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        def mock_embed(text, model=None):
            idx = docs.index(text) if text in docs else 99
            v = [0.0] * 4
            v[idx % 4] = 1.0
            return v

        with patch("embeddings_demo.get_embedding", side_effect=mock_embed), \
             patch("embeddings_demo.batch_get_embeddings", side_effect=lambda texts, **kw: [mock_embed(t) for t in texts]):
            results = semantic_search("doc1", docs, top_k=3)

        assert len(results) <= 3

    def test_results_sorted_by_score_descending(self, fake_openai_key):
        docs = ["apple", "banana", "cherry"]

        # Query vector points in direction of "apple"
        def mock_embed(text, model=None):
            if text == "apple":
                return [1.0, 0.0, 0.0, 0.0]
            if text == "banana":
                return [0.7, 0.3, 0.0, 0.0]
            return [0.1, 0.1, 0.8, 0.0]  # cherry or query

        with patch("embeddings_demo.get_embedding", side_effect=mock_embed), \
             patch("embeddings_demo.batch_get_embeddings", side_effect=lambda texts, **kw: [mock_embed(t) for t in texts]):
            results = semantic_search("apple", docs, top_k=3)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_dicts_have_required_keys(self, fake_openai_key):
        docs = ["doc one", "doc two"]

        def mock_embed(text, model=None):
            return [0.5, 0.5, 0.0, 0.0]

        with patch("embeddings_demo.get_embedding", side_effect=mock_embed), \
             patch("embeddings_demo.batch_get_embeddings", side_effect=lambda texts, **kw: [mock_embed(t) for t in texts]):
            results = semantic_search("query", docs, top_k=2)

        for r in results:
            assert "rank" in r
            assert "score" in r
            assert "text" in r
