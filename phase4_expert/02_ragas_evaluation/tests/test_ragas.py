"""Tests for ragas_eval.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ragas_eval import (
    SAMPLE_EVAL_DATA,
    create_ragas_dataset,
    interpret_scores,
    run_ragas_evaluation,
)


class TestCreateRagasDataset:
    def test_creates_dataset_with_correct_length(self):
        dataset = create_ragas_dataset(SAMPLE_EVAL_DATA)
        assert len(dataset) == len(SAMPLE_EVAL_DATA)

    def test_dataset_has_required_columns(self):
        dataset = create_ragas_dataset(SAMPLE_EVAL_DATA)
        for col in ["question", "answer", "contexts", "ground_truth"]:
            assert col in dataset.column_names

    def test_contexts_column_contains_lists(self):
        dataset = create_ragas_dataset(SAMPLE_EVAL_DATA)
        for row in dataset:
            assert isinstance(row["contexts"], list)
            assert len(row["contexts"]) > 0

    def test_empty_list_creates_empty_dataset(self):
        dataset = create_ragas_dataset([])
        assert len(dataset) == 0

    def test_single_sample(self):
        data = [{
            "question": "test?",
            "answer": "answer",
            "contexts": ["context"],
            "ground_truth": "ground truth",
        }]
        dataset = create_ragas_dataset(data)
        assert len(dataset) == 1
        assert dataset[0]["question"] == "test?"


class TestRunRagasEvaluation:
    def test_raises_without_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_dataset = MagicMock()
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            run_ragas_evaluation(mock_dataset)

    def test_returns_dict_with_metric_scores(self, fake_openai_key):
        dataset = create_ragas_dataset(SAMPLE_EVAL_DATA[:2])

        mock_results = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.90,
            "context_precision": 0.75,
            "context_recall": 0.80,
        }

        with patch("ragas.evaluate") as mock_evaluate, \
             patch("ragas.metrics.faithfulness") as mock_faith, \
             patch("ragas.metrics.answer_relevancy") as mock_ar, \
             patch("ragas.metrics.context_precision") as mock_cp, \
             patch("ragas.metrics.context_recall") as mock_cr:

            mock_faith.name = "faithfulness"
            mock_ar.name = "answer_relevancy"
            mock_cp.name = "context_precision"
            mock_cr.name = "context_recall"

            mock_evaluate.return_value = mock_results

            result = run_ragas_evaluation(dataset, metrics=[mock_faith, mock_ar])

        assert isinstance(result, dict)


class TestInterpretScores:
    def test_good_scores_get_good_level(self):
        scores = {"faithfulness": 0.90, "answer_relevancy": 0.85}
        result = interpret_scores(scores)
        assert result["faithfulness"]["level"] == "good"
        assert result["answer_relevancy"]["level"] == "good"

    def test_bad_scores_get_bad_level(self):
        scores = {"faithfulness": 0.30, "context_precision": 0.40}
        result = interpret_scores(scores)
        assert result["faithfulness"]["level"] == "bad"
        assert result["context_precision"]["level"] == "bad"

    def test_fair_scores_get_fair_level(self):
        scores = {"faithfulness": 0.70}
        result = interpret_scores(scores)
        assert result["faithfulness"]["level"] == "fair"

    def test_each_metric_has_recommendation(self):
        scores = {"faithfulness": 0.30, "answer_relevancy": 0.90}
        result = interpret_scores(scores)
        assert result["faithfulness"]["recommendation"] != ""

    def test_result_has_expected_keys(self):
        scores = {"faithfulness": 0.75}
        result = interpret_scores(scores)
        assert "score" in result["faithfulness"]
        assert "level" in result["faithfulness"]
        assert "emoji" in result["faithfulness"]
        assert "recommendation" in result["faithfulness"]

    def test_good_score_has_checkmark_emoji(self):
        scores = {"faithfulness": 0.95}
        result = interpret_scores(scores)
        assert result["faithfulness"]["emoji"] == "✓"

    def test_bad_score_has_cross_emoji(self):
        scores = {"faithfulness": 0.20}
        result = interpret_scores(scores)
        assert result["faithfulness"]["emoji"] == "✗"

    def test_unknown_metric_gets_default_thresholds(self):
        scores = {"custom_metric": 0.50}
        result = interpret_scores(scores)
        assert "custom_metric" in result
