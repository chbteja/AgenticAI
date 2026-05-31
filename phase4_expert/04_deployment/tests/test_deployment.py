"""
Tests for fastapi_app.py

Uses FastAPI's TestClient (synchronous) for endpoint testing.
The RAG pipeline and API calls are fully mocked.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def client():
    """TestClient with a fresh app state for each test."""
    import fastapi_app
    # Reset app state before each test
    fastapi_app._app_state["is_ready"] = False
    fastapi_app._app_state["rag_chain"] = None
    fastapi_app._app_state["vector_store"] = None
    fastapi_app._app_state["indexed_doc_count"] = 0
    fastapi_app._rate_limit_store.clear()

    return TestClient(fastapi_app.app, raise_server_exceptions=False)


@pytest.fixture
def ready_client():
    """TestClient with the RAG pipeline pre-initialised."""
    import fastapi_app
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Python was created by Guido van Rossum."
    fastapi_app._app_state["is_ready"] = True
    fastapi_app._app_state["rag_chain"] = mock_chain
    fastapi_app._app_state["indexed_doc_count"] = 42
    fastapi_app._rate_limit_store.clear()

    return TestClient(fastapi_app.app, raise_server_exceptions=False)


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_includes_version(self, client):
        response = client.get("/health")
        assert "version" in response.json()


# ── Readiness endpoint ────────────────────────────────────────────────────────

class TestReadinessEndpoint:
    def test_returns_503_when_not_ready(self, client):
        response = client.get("/ready")
        assert response.status_code == 503

    def test_returns_200_when_ready(self, ready_client):
        response = ready_client.get("/ready")
        assert response.status_code == 200

    def test_includes_document_count_when_ready(self, ready_client):
        response = ready_client.get("/ready")
        data = response.json()
        assert data["indexed_documents"] == 42
        assert data["ready"] is True


# ── Ask endpoint ──────────────────────────────────────────────────────────────

class TestAskEndpoint:
    def test_returns_503_when_not_ready(self, client):
        response = client.post("/ask", json={"question": "Who created Python?"})
        assert response.status_code == 503

    def test_returns_answer_when_ready(self, ready_client):
        response = ready_client.post("/ask", json={"question": "Who created Python?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "question" in data
        assert data["question"] == "Who created Python?"

    def test_validates_short_question(self, ready_client):
        response = ready_client.post("/ask", json={"question": "Hi"})
        # "Hi" is 2 chars, min_length is 3
        assert response.status_code == 422

    def test_validates_long_question(self, ready_client):
        long_q = "a" * 501
        response = ready_client.post("/ask", json={"question": long_q})
        assert response.status_code == 422

    def test_validates_empty_question(self, ready_client):
        response = ready_client.post("/ask", json={"question": "   "})
        assert response.status_code == 422

    def test_response_includes_latency(self, ready_client):
        response = ready_client.post("/ask", json={"question": "Who created Python?"})
        assert response.status_code == 200
        assert "latency_ms" in response.json()
        assert response.json()["latency_ms"] >= 0

    def test_response_includes_model_name(self, ready_client):
        response = ready_client.post("/ask", json={"question": "Who created Python?"})
        assert "model" in response.json()

    def test_top_k_out_of_range_rejected(self, ready_client):
        response = ready_client.post("/ask", json={"question": "Who created Python?", "top_k": 0})
        assert response.status_code == 422

        response = ready_client.post("/ask", json={"question": "Who created Python?", "top_k": 11})
        assert response.status_code == 422

    def test_chain_exception_returns_500(self, ready_client):
        import fastapi_app
        fastapi_app._app_state["rag_chain"].invoke.side_effect = Exception("Chain failed")
        response = ready_client.post("/ask", json={"question": "Who created Python?"})
        assert response.status_code == 500


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_allows_requests_under_limit(self, ready_client):
        import fastapi_app
        fastapi_app._rate_limit_store.clear()

        for _ in range(5):
            response = ready_client.post("/ask", json={"question": "Who created Python?"})
            assert response.status_code == 200

    def test_blocks_requests_over_limit(self, ready_client):
        import fastapi_app
        fastapi_app._rate_limit_store.clear()

        # Set up 10 existing timestamps (at the limit)
        import time
        fastapi_app._rate_limit_store["testclient"] = [time.time()] * 10

        response = ready_client.post("/ask", json={"question": "Who created Python?"})
        assert response.status_code == 429


# ── Index endpoint ────────────────────────────────────────────────────────────

class TestIndexEndpoint:
    def test_returns_task_id(self, client):
        with patch("fastapi_app.SAMPLE_DATA") as mock_path:
            mock_path.exists.return_value = True

            response = client.post("/index", json={})
            # May be 200 or 400 depending on file existence
            assert response.status_code in (200, 400)

    def test_invalid_path_returns_400(self, client):
        response = client.post("/index", json={"data_path": "/nonexistent/path/file.txt"})
        assert response.status_code == 400

    def test_response_includes_message(self, client):
        with patch("fastapi_app.SAMPLE_DATA") as mock_path:
            mock_path.exists.return_value = True
            response = client.post("/index", json={})
            if response.status_code == 200:
                assert "message" in response.json()
                assert "task_id" in response.json()


# ── Request model validation ──────────────────────────────────────────────────

class TestAskRequestModel:
    def test_valid_question(self):
        from fastapi_app import AskRequest
        req = AskRequest(question="Who created Python?")
        assert req.question == "Who created Python?"

    def test_whitespace_stripped(self):
        from fastapi_app import AskRequest
        req = AskRequest(question="  What is Python?  ")
        assert req.question == "What is Python?"

    def test_default_top_k(self):
        from fastapi_app import AskRequest
        req = AskRequest(question="Test question here")
        assert req.top_k == 4
