"""Tests for llm_judge.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_judge import (
    EvaluationResult,
    EvaluationScore,
    _build_judge_prompt,
    _parse_score_from_response,
    batch_evaluate,
    evaluate_full,
)


class TestEvaluationScore:
    def test_valid_score_created(self):
        score = EvaluationScore(dimension="faithfulness", score=4, reasoning="Good grounding.")
        assert score.score == 4
        assert score.dimension == "faithfulness"

    def test_score_below_range_raises(self):
        with pytest.raises(Exception):
            EvaluationScore(dimension="faithfulness", score=0, reasoning="test")

    def test_score_above_range_raises(self):
        with pytest.raises(Exception):
            EvaluationScore(dimension="faithfulness", score=6, reasoning="test")

    def test_score_5_valid(self):
        score = EvaluationScore(dimension="relevance", score=5, reasoning="Perfect")
        assert score.score == 5


class TestEvaluationResult:
    def test_overall_score_is_mean(self):
        scores = [
            EvaluationScore(dimension="faithfulness", score=4, reasoning="good"),
            EvaluationScore(dimension="relevance", score=2, reasoning="ok"),
        ]
        result = EvaluationResult(question="q", context="c", answer="a", scores=scores)
        assert result.overall_score == 3.0

    def test_to_dict_has_expected_keys(self):
        scores = [EvaluationScore(dimension="faithfulness", score=5, reasoning="perfect")]
        result = EvaluationResult(question="q", context="c", answer="a", scores=scores)
        d = result.to_dict()
        assert "question" in d
        assert "answer" in d
        assert "scores" in d
        assert "overall" in d

    def test_empty_scores_gives_zero_overall(self):
        result = EvaluationResult(question="q", context="c", answer="a", scores=[])
        assert result.overall_score == 0.0


class TestParseScoreFromResponse:
    def test_parses_valid_score_and_reason(self):
        response = "Score: 4\nReason: The answer is mostly grounded in context."
        result = _parse_score_from_response(response, "faithfulness")
        assert result is not None
        assert result.score == 4
        assert "grounded" in result.reasoning

    def test_returns_none_for_no_score(self):
        result = _parse_score_from_response("No score here", "faithfulness")
        assert result is None

    def test_handles_uppercase_score(self):
        response = "SCORE: 3\nREASON: Mediocre."
        result = _parse_score_from_response(response, "relevance")
        assert result is not None
        assert result.score == 3

    def test_extracts_score_5(self):
        response = "Score: 5\nReason: Perfect."
        result = _parse_score_from_response(response, "coherence")
        assert result.score == 5

    def test_extracts_score_1(self):
        response = "Score: 1\nReason: Completely wrong."
        result = _parse_score_from_response(response, "faithfulness")
        assert result.score == 1


class TestBuildJudgePrompt:
    def test_prompt_contains_question(self):
        prompt = _build_judge_prompt(
            question="What is Python?",
            context="Python is a language.",
            answer="Python is a language.",
            dimension="faithfulness",
            description="Is it faithful?",
            rubric={1: "bad", 5: "perfect"},
        )
        assert "What is Python?" in prompt

    def test_prompt_contains_context(self):
        prompt = _build_judge_prompt(
            question="q", context="specific context here", answer="a",
            dimension="faithfulness", description="desc", rubric={1: "bad", 5: "good"},
        )
        assert "specific context here" in prompt

    def test_prompt_contains_dimension(self):
        prompt = _build_judge_prompt(
            question="q", context="c", answer="a",
            dimension="coherence", description="desc", rubric={1: "bad", 5: "good"},
        )
        assert "coherence" in prompt.lower()

    def test_prompt_requests_score_format(self):
        prompt = _build_judge_prompt(
            question="q", context="c", answer="a",
            dimension="relevance", description="desc", rubric={1: "bad", 5: "good"},
        )
        assert "Score:" in prompt


class TestEvaluateFull:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            evaluate_full("q", "c", "a")

    def test_returns_evaluation_result(self, fake_anthropic_key):
        mock_score = EvaluationScore(dimension="faithfulness", score=4, reasoning="Good.")
        with patch("llm_judge.score_single_dimension", return_value=mock_score):
            result = evaluate_full("q", "c", "a", dimensions=["faithfulness"])
        assert isinstance(result, EvaluationResult)
        assert len(result.scores) == 1

    def test_uses_all_dimensions_by_default(self, fake_anthropic_key):
        mock_score = EvaluationScore(dimension="x", score=3, reasoning="ok.")
        with patch("llm_judge.score_single_dimension", return_value=mock_score) as mock_fn:
            evaluate_full("q", "c", "a")
        assert mock_fn.call_count == 4  # 4 dimensions

    def test_skips_none_scores(self, fake_anthropic_key):
        with patch("llm_judge.score_single_dimension", return_value=None):
            result = evaluate_full("q", "c", "a", dimensions=["faithfulness"])
        assert len(result.scores) == 0


class TestBatchEvaluate:
    def test_returns_expected_keys(self, fake_anthropic_key):
        mock_score = EvaluationScore(dimension="faithfulness", score=4, reasoning="Good.")
        with patch("llm_judge.score_single_dimension", return_value=mock_score):
            result = batch_evaluate([{"question": "q", "context": "c", "answer": "a"}])
        assert "results" in result
        assert "aggregate_scores" in result
        assert "overall_mean" in result
        assert "weakest_dimension" in result

    def test_processes_all_cases(self, fake_anthropic_key):
        cases = [
            {"question": f"q{i}", "context": "c", "answer": "a"} for i in range(3)
        ]
        mock_score = EvaluationScore(dimension="faithfulness", score=3, reasoning="ok.")
        with patch("llm_judge.score_single_dimension", return_value=mock_score):
            result = batch_evaluate(cases)
        assert len(result["results"]) == 3
