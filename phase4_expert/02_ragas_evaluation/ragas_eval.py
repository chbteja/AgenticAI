"""
ragas_eval.py — RAG Evaluation with RAGAS

Purpose:
    Use the RAGAS framework to evaluate RAG pipeline quality across four dimensions:
    faithfulness, answer relevancy, context precision, and context recall.

Learning Objectives:
    1. Understand what each RAGAS metric measures and which failure mode it targets.
    2. Create a RAGAS evaluation dataset from RAG pipeline outputs.
    3. Run RAGAS evaluation and interpret scores.
    4. Use low scores to guide RAG improvements.

Tech Stack: ragas, langchain, langchain-openai, datasets
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Sample evaluation dataset ─────────────────────────────────────────────────
# In production, generate this by running your RAG pipeline on a test question set.
# Each sample needs: question, answer (from RAG), contexts (retrieved chunks), ground_truth.

SAMPLE_EVAL_DATA = [
    {
        "question": "Who created Python?",
        "answer": "Python was created by Guido van Rossum.",
        "contexts": [
            "Python is a high-level interpreted language created by Guido van Rossum. "
            "It was first released in 1991 while he was working at CWI in Amsterdam.",
            "Python's design philosophy emphasises code readability.",
        ],
        "ground_truth": "Python was created by Guido van Rossum and first released in 1991.",
    },
    {
        "question": "What are Python's main web frameworks?",
        "answer": "Django, Flask, and FastAPI are Python's main web frameworks.",
        "contexts": [
            "Django and Flask are the two most popular Python web frameworks. "
            "Django is batteries-included. Flask is a micro-framework. FastAPI is newer.",
        ],
        "ground_truth": "Django, Flask, and FastAPI are the main Python web frameworks.",
    },
    {
        "question": "What is the Zen of Python?",
        "answer": "The Zen of Python is a set of aphorisms that includes 'Beautiful is better than ugly' and 'Readability counts'.",
        "contexts": [
            "Python's core philosophy is summarised in 'The Zen of Python', "
            "which includes aphorisms such as: Beautiful is better than ugly. "
            "Explicit is better than implicit. Simple is better than complex. "
            "Readability counts.",
        ],
        "ground_truth": "The Zen of Python is a set of guiding aphorisms for Python design.",
    },
    {
        "question": "When did Python 2 reach end-of-life?",
        "answer": "Python 2 reached end-of-life on January 1, 2020.",
        "contexts": [
            "Python 2 reached end-of-life on January 1, 2020. "
            "All new projects should use Python 3.",
        ],
        "ground_truth": "Python 2 reached end-of-life on January 1, 2020.",
    },
    {
        "question": "What is the speed of light?",  # Out-of-scope — tests faithfulness
        "answer": "The speed of light is approximately 299,792,458 metres per second.",
        "contexts": [
            "Python was created by Guido van Rossum in 1991.",  # Irrelevant context
        ],
        "ground_truth": "The speed of light is approximately 299,792,458 metres per second.",
    },
]


def create_ragas_dataset(eval_data: list[dict]):
    """
    Convert evaluation data list into a RAGAS-compatible HuggingFace Dataset.

    RAGAS requires:
        - question: str
        - answer: str
        - contexts: list[str]
        - ground_truth: str (for context recall)

    Args:
        eval_data: List of dicts with the above keys.

    Returns:
        HuggingFace Dataset object.
    """
    logger.info("create_ragas_dataset: %d samples", len(eval_data))

    from datasets import Dataset

    dataset = Dataset.from_list(eval_data)
    logger.info("Dataset created: %d rows, columns=%s", len(dataset), dataset.column_names)
    return dataset


def run_ragas_evaluation(
    dataset,
    metrics: list | None = None,
) -> dict:
    """
    Run RAGAS evaluation on the dataset and return scores.

    Args:
        dataset: HuggingFace Dataset with required columns.
        metrics: List of RAGAS metric objects. Defaults to all 4 core metrics.

    Returns:
        Dict of metric_name → score (float 0–1).

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set (RAGAS uses OpenAI by default).
    """
    logger.info("run_ragas_evaluation: %d samples", len(dataset))

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set — RAGAS requires OpenAI for evaluation LLM")

    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        logger.info("Using all 4 RAGAS metrics")

    results = evaluate(dataset, metrics=metrics)
    scores = {str(metric.name): float(results[metric.name]) for metric in metrics}

    logger.info("RAGAS evaluation complete: %s", scores)
    return scores


def interpret_scores(scores: dict) -> dict:
    """
    Interpret RAGAS scores and provide actionable recommendations.

    Args:
        scores: Dict of metric_name → float score (0–1).

    Returns:
        Dict of metric_name → {score, interpretation, recommendation}.
    """
    logger.info("interpret_scores: %s", scores)

    thresholds = {
        "faithfulness": {
            "good": 0.80,
            "fair": 0.60,
            "recommendations": {
                "bad": "Your generator is hallucinating. Check that your RAG prompt explicitly says to use only the context.",
                "fair": "Some hallucination detected. Consider strengthening the system prompt grounding instruction.",
                "good": "Good faithfulness. The generator stays grounded in retrieved context.",
            },
        },
        "answer_relevancy": {
            "good": 0.80,
            "fair": 0.60,
            "recommendations": {
                "bad": "Answers are off-topic. Review your RAG prompt and check if the question format is ambiguous.",
                "fair": "Answers are partially relevant. Consider adding 'answer the question directly' to your prompt.",
                "good": "Good relevancy. Answers address the questions asked.",
            },
        },
        "context_precision": {
            "good": 0.75,
            "fair": 0.55,
            "recommendations": {
                "bad": "Retriever returns many irrelevant chunks. Try smaller chunk size, better reranking, or stricter retrieval.",
                "fair": "Some irrelevant chunks retrieved. Consider adding reranking (Phase 3, module 04).",
                "good": "Good precision. Retrieved chunks are relevant.",
            },
        },
        "context_recall": {
            "good": 0.75,
            "fair": 0.55,
            "recommendations": {
                "bad": "Retriever misses important information. Try HyDE or multi-query retrieval (Phase 3, modules 01-02).",
                "fair": "Some information is missed. Consider increasing top-k or trying multi-query.",
                "good": "Good recall. Retriever finds the necessary information.",
            },
        },
    }

    interpretations = {}
    for metric, score in scores.items():
        config = thresholds.get(metric, {"good": 0.75, "fair": 0.55, "recommendations": {}})
        if score >= config["good"]:
            level = "good"
            emoji = "✓"
        elif score >= config["fair"]:
            level = "fair"
            emoji = "~"
        else:
            level = "bad"
            emoji = "✗"

        interpretations[metric] = {
            "score": round(score, 3),
            "level": level,
            "emoji": emoji,
            "recommendation": config["recommendations"].get(level, ""),
        }

    return interpretations


def generate_eval_dataset_from_rag(
    questions: list[str],
    rag_chain,
    vector_store,
) -> list[dict]:
    """
    Run a list of questions through a RAG pipeline and collect outputs for RAGAS.

    Use this to create an evaluation dataset from your actual RAG system
    rather than using pre-built sample data.

    Args:
        questions:    List of question strings.
        rag_chain:    A LangChain chain that takes {"question": str} and returns str.
        vector_store: The vector store used for retrieval.

    Returns:
        List of dicts ready for create_ragas_dataset().
    """
    logger.info("generate_eval_dataset_from_rag: %d questions", len(questions))

    eval_data = []
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    for q in questions:
        logger.debug("Processing question: %r", q[:60])
        docs = retriever.invoke(q)
        contexts = [d.page_content for d in docs]
        answer = rag_chain.invoke({"question": q})
        eval_data.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": "",  # Must be filled in manually or with a reference system
        })

    logger.info("Generated %d evaluation samples", len(eval_data))
    return eval_data


def main() -> None:
    logger.info("=== RAGAS Evaluation Demo Starting ===")

    # ── Create dataset ─────────────────────────────────────────────────────────
    print("\n=== Creating RAGAS Dataset ===")
    dataset = create_ragas_dataset(SAMPLE_EVAL_DATA)
    print(f"Dataset: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")

    # ── Run evaluation ─────────────────────────────────────────────────────────
    print("\n=== Running RAGAS Evaluation ===")
    print("(This makes API calls to OpenAI — requires OPENAI_API_KEY)")

    if not os.environ.get("OPENAI_API_KEY"):
        print("[!] OPENAI_API_KEY not set. Showing sample scores for demonstration.\n")
        scores = {
            "faithfulness": 0.82,
            "answer_relevancy": 0.91,
            "context_precision": 0.68,
            "context_recall": 0.73,
        }
        print("(Sample scores — run with your API key for real evaluation)")
    else:
        scores = run_ragas_evaluation(dataset)

    # ── Interpret results ──────────────────────────────────────────────────────
    print("\n=== Score Interpretation ===")
    interpretations = interpret_scores(scores)
    for metric, info in interpretations.items():
        print(f"\n  {info['emoji']} {metric}: {info['score']:.3f} ({info['level'].upper()})")
        if info["recommendation"]:
            print(f"     → {info['recommendation']}")

    logger.info("=== RAGAS Evaluation Demo Complete ===")


if __name__ == "__main__":
    main()
