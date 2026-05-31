"""
llm_judge.py — LLM-as-a-Judge Evaluation

Purpose:
    Use a separate LLM to score RAG system outputs on multiple quality dimensions.
    This gives you a quantitative, scalable evaluation signal.

Learning Objectives:
    1. Design a multi-dimension evaluation rubric.
    2. Write judge prompts that extract reliable numeric scores.
    3. Parse structured scores from LLM responses.
    4. Run batch evaluation and aggregate results.
    5. Identify the weakest dimension to guide improvement.

Security: Judge scores are parsed via regex, not eval — safe from prompt injection.

Tech Stack: anthropic, pydantic, python-dotenv
"""

import logging
import os
import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-sonnet-4-6"  # Use a strong model for judging
GEN_MODEL = "claude-haiku-4-5-20251001"  # Cheaper model for generation


# ── Data models ────────────────────────────────────────────────────────────────

class EvaluationScore(BaseModel):
    """Structured scores for one dimension of evaluation."""

    dimension: str
    score: int = Field(ge=1, le=5)
    reasoning: str

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError(f"Score must be 1–5, got {v}")
        return v


@dataclass
class EvaluationResult:
    """Complete evaluation result for one question-answer-context triple."""

    question: str
    context: str
    answer: str
    scores: list[EvaluationScore] = field(default_factory=list)
    overall_score: float = 0.0

    def __post_init__(self) -> None:
        if self.scores:
            self.overall_score = mean(s.score for s in self.scores)

    def to_dict(self) -> dict:
        return {
            "question": self.question[:80],
            "answer": self.answer[:80],
            "scores": {s.dimension: s.score for s in self.scores},
            "overall": round(self.overall_score, 2),
        }


# ── Judge prompt builders ──────────────────────────────────────────────────────

def _build_judge_prompt(
    question: str,
    context: str,
    answer: str,
    dimension: str,
    description: str,
    rubric: dict[int, str],
) -> str:
    """Build a judge prompt for a single evaluation dimension."""
    rubric_text = "\n".join(f"  {score}: {desc}" for score, desc in sorted(rubric.items()))
    return f"""You are an expert evaluator. Score the following answer on the dimension of {dimension}.

Question: {question}

Context provided to the system:
{context}

System answer:
{answer}

Dimension: {dimension}
{description}

Scoring rubric:
{rubric_text}

Instructions:
1. Think carefully about whether the answer meets the criteria.
2. Assign a score from 1 to 5.
3. Give a brief one-sentence justification.

Respond in EXACTLY this format:
Score: <integer 1-5>
Reason: <one sentence>"""


DIMENSIONS: dict[str, dict] = {
    "faithfulness": {
        "description": "Does the answer make claims that are supported by the provided context? "
                       "Penalise any claims not found in or inferable from the context.",
        "rubric": {
            1: "Answer contradicts or invents information not in context",
            2: "Answer mixes context facts with outside knowledge",
            3: "Answer mostly uses context but with one unsupported claim",
            4: "Answer is almost entirely grounded in context",
            5: "Every claim in the answer is directly supported by the context",
        },
    },
    "relevance": {
        "description": "Does the answer actually address the question asked?",
        "rubric": {
            1: "Answer is completely off-topic",
            2: "Answer touches the topic but misses the question",
            3: "Answer partially addresses the question",
            4: "Answer addresses the question with minor gaps",
            5: "Answer directly and fully addresses the question",
        },
    },
    "completeness": {
        "description": "Does the answer include all important information available in the context?",
        "rubric": {
            1: "Answer misses most of the relevant context information",
            2: "Answer misses more than half of relevant information",
            3: "Answer captures about half of relevant information",
            4: "Answer is mostly complete with minor omissions",
            5: "Answer captures all key information from the context",
        },
    },
    "coherence": {
        "description": "Is the answer well-structured, clear, and easy to read?",
        "rubric": {
            1: "Answer is incoherent or incomprehensible",
            2: "Answer is hard to follow with major clarity issues",
            3: "Answer is understandable but poorly structured",
            4: "Answer is clear with minor formatting issues",
            5: "Answer is exceptionally clear, well-structured, and concise",
        },
    },
}


# ── Scoring functions ──────────────────────────────────────────────────────────

def _parse_score_from_response(response: str, dimension: str) -> Optional[EvaluationScore]:
    """
    Parse a score and reasoning from the judge's text response.

    Uses regex to extract the score — safe from prompt injection since
    we never evaluate the response as code.

    Args:
        response:  Raw text from the judge LLM.
        dimension: Which dimension was being scored.

    Returns:
        EvaluationScore or None if parsing fails.
    """
    logger.debug("_parse_score_from_response: dimension=%s, response_len=%d", dimension, len(response))

    score_match = re.search(r"Score:\s*([1-5])", response, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)

    if not score_match:
        logger.warning("Could not parse score from response: %r", response[:100])
        return None

    score = int(score_match.group(1))
    reasoning = reason_match.group(1).strip() if reason_match else "No reasoning provided"

    logger.debug("Parsed score: dimension=%s, score=%d", dimension, score)
    return EvaluationScore(dimension=dimension, score=score, reasoning=reasoning)


def score_single_dimension(
    question: str,
    context: str,
    answer: str,
    dimension: str,
) -> Optional[EvaluationScore]:
    """
    Score a single dimension for one question-context-answer triple.

    Args:
        question:  The question that was asked.
        context:   The context that was provided to the generator.
        answer:    The generator's answer.
        dimension: One of: faithfulness, relevance, completeness, coherence.

    Returns:
        EvaluationScore or None if the dimension is unknown or parsing fails.
    """
    logger.info("score_single_dimension: dimension=%s", dimension)

    if dimension not in DIMENSIONS:
        logger.error("Unknown dimension: %r. Valid: %s", dimension, list(DIMENSIONS.keys()))
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    dim_config = DIMENSIONS[dimension]
    prompt = _build_judge_prompt(
        question=question,
        context=context,
        answer=answer,
        dimension=dimension,
        description=dim_config["description"],
        rubric=dim_config["rubric"],
    )

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text
    logger.debug("Judge raw response: %r", raw_text[:150])

    return _parse_score_from_response(raw_text, dimension)


def evaluate_full(
    question: str,
    context: str,
    answer: str,
    dimensions: Optional[list[str]] = None,
) -> EvaluationResult:
    """
    Evaluate an answer across all (or specified) dimensions.

    Args:
        question:   The question asked.
        context:    The context provided to the generator.
        answer:     The generator's answer to evaluate.
        dimensions: List of dimension names. Defaults to all 4 dimensions.

    Returns:
        EvaluationResult with scores for all dimensions.
    """
    logger.info("evaluate_full: question=%r, dimensions=%s", question[:60], dimensions)

    if dimensions is None:
        dimensions = list(DIMENSIONS.keys())

    scores: list[EvaluationScore] = []
    for dim in dimensions:
        score = score_single_dimension(question, context, answer, dim)
        if score:
            scores.append(score)
        else:
            logger.warning("Skipping dimension %r — no score returned", dim)

    result = EvaluationResult(
        question=question,
        context=context,
        answer=answer,
        scores=scores,
    )
    logger.info(
        "evaluate_full: overall=%.2f, dimensions=%d",
        result.overall_score,
        len(scores),
    )
    return result


def batch_evaluate(test_cases: list[dict]) -> dict:
    """
    Run evaluation across a batch of test cases and aggregate results.

    Args:
        test_cases: List of dicts with keys: question, context, answer.

    Returns:
        Dict with keys: results (list), aggregate_scores (per dimension), overall_mean.
    """
    logger.info("batch_evaluate: %d test cases", len(test_cases))

    results: list[EvaluationResult] = []
    for i, case in enumerate(test_cases):
        logger.info("Evaluating case %d/%d: %r", i + 1, len(test_cases), case["question"][:40])
        result = evaluate_full(
            question=case["question"],
            context=case["context"],
            answer=case["answer"],
        )
        results.append(result)

    # Aggregate scores by dimension
    aggregate: dict[str, list[int]] = {dim: [] for dim in DIMENSIONS}
    for result in results:
        for score in result.scores:
            if score.dimension in aggregate:
                aggregate[score.dimension].append(score.score)

    aggregate_means = {
        dim: round(mean(scores), 2) if scores else 0.0
        for dim, scores in aggregate.items()
    }

    overall_mean = round(mean(r.overall_score for r in results), 2)
    logger.info("batch_evaluate done: overall_mean=%.2f", overall_mean)

    return {
        "results": [r.to_dict() for r in results],
        "aggregate_scores": aggregate_means,
        "overall_mean": overall_mean,
        "weakest_dimension": min(aggregate_means, key=aggregate_means.get) if aggregate_means else None,
    }


# ── Demo ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== LLM-as-Judge Demo Starting ===")

    test_cases = [
        {
            "question": "Who created Python?",
            "context": "Python was created by Guido van Rossum and first released in 1991. "
                       "Van Rossum was employed at CWI in Amsterdam when he developed Python.",
            "answer": "Python was created by Guido van Rossum.",
        },
        {
            "question": "What Python web frameworks exist?",
            "context": "Django and Flask are the two most popular Python web frameworks. "
                       "Django is batteries-included. Flask is a micro-framework. FastAPI is newer.",
            "answer": "Django, Flask, and FastAPI are Python web frameworks.",
        },
        {
            "question": "What is the capital of France?",
            "context": "Python was created by Guido van Rossum in 1991.",  # Context is irrelevant!
            "answer": "The capital of France is Paris.",  # Hallucinated from training data
        },
    ]

    print("\n=== Batch Evaluation ===")
    summary = batch_evaluate(test_cases)

    print("\n--- Per-case results ---")
    for r in summary["results"]:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer']}")
        for dim, score in r["scores"].items():
            print(f"  {dim:15s}: {score}/5")
        print(f"  Overall:         {r['overall']}/5")

    print("\n--- Aggregate summary ---")
    for dim, score in summary["aggregate_scores"].items():
        bar = "█" * int(score) + "░" * (5 - int(score))
        print(f"  {dim:15s}: {bar} {score}/5")
    print(f"\n  Overall mean:  {summary['overall_mean']}/5")
    print(f"  Weakest area:  {summary['weakest_dimension']}")

    logger.info("=== LLM-as-Judge Demo Complete ===")


if __name__ == "__main__":
    main()
