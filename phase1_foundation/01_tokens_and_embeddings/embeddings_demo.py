"""
embeddings_demo.py — Vector Embeddings and Semantic Search

Purpose:
    Show how text is converted into high-dimensional float vectors (embeddings)
    and how those vectors encode semantic meaning. Build a minimal semantic search
    function from scratch using only the OpenAI embeddings API and numpy.

Learning Objectives:
    1. Generate embeddings via the OpenAI API.
    2. Understand cosine similarity as a semantic distance metric.
    3. Build a semantic search function without a vector database.
    4. Observe that semantically similar sentences cluster together in vector space.

Tech Stack: openai, python-dotenv, (numpy for cosine similarity)
"""

import logging
import math
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Embedding model — text-embedding-3-small is fast and cheap; good for learning
EMBEDDING_MODEL = "text-embedding-3-small"


def _get_openai_client():
    """Create an OpenAI client, raising clearly if the key is missing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file. "
            "See .env.example for the template."
        )
    from openai import OpenAI
    logger.debug("OpenAI client initialized")
    return OpenAI(api_key=api_key)


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Fetch a vector embedding for *text* from the OpenAI API.

    The vector has 1536 dimensions for text-embedding-3-small.
    Each dimension captures some latent semantic feature.

    Args:
        text:  The text to embed. Must be non-empty.
        model: OpenAI embedding model name.

    Returns:
        List of floats representing the embedding vector.

    Raises:
        ValueError: If *text* is empty.
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    logger.debug("get_embedding called: model=%s, text_len=%d", model, len(text))

    if not text or not text.strip():
        raise ValueError("Cannot embed empty or whitespace-only text")

    client = _get_openai_client()
    response = client.embeddings.create(input=text, model=model)
    vector = response.data[0].embedding

    logger.info(
        "Embedding generated: model=%s, dimensions=%d, text=%r",
        model,
        len(vector),
        text[:50],
    )
    return vector


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute the cosine similarity between two embedding vectors.

    Cosine similarity measures the angle between vectors, not their magnitude.
    This makes it robust to texts of different lengths.

    Returns:
        Float in [-1.0, 1.0].
        1.0  = identical direction (semantically identical)
        0.0  = orthogonal (unrelated)
        -1.0 = opposite direction (semantically opposite)

    Raises:
        ValueError: If vectors have different lengths or are zero vectors.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cannot compute cosine similarity with zero vectors")

    similarity = dot_product / (magnitude_a * magnitude_b)
    # Clamp to [-1, 1] to handle floating point rounding errors
    return max(-1.0, min(1.0, similarity))


def batch_get_embeddings(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    Embed multiple texts in a single API call (more efficient than looping).

    Args:
        texts: Non-empty list of strings. Each must be non-empty.
        model: OpenAI embedding model name.

    Returns:
        List of embedding vectors, one per input text, in the same order.

    Raises:
        ValueError: If *texts* is empty or contains empty strings.
    """
    logger.debug("batch_get_embeddings called: %d texts", len(texts))

    if not texts:
        raise ValueError("texts list must not be empty")

    for i, t in enumerate(texts):
        if not t or not t.strip():
            raise ValueError(f"Text at index {i} is empty — all texts must be non-empty")

    client = _get_openai_client()
    response = client.embeddings.create(input=texts, model=model)

    # The API returns results in the same order as the input
    vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    logger.info("Batch embeddings: %d texts → %d vectors (dim=%d)", len(texts), len(vectors), len(vectors[0]))
    return vectors


def semantic_search(
    query: str,
    documents: list[str],
    top_k: int = 3,
) -> list[dict]:
    """
    Find the *top_k* most semantically similar documents to *query*.

    This is a minimal implementation of semantic search:
    1. Embed the query.
    2. Embed all documents (batch call for efficiency).
    3. Compute cosine similarity between query and each document.
    4. Return top-k results sorted by similarity descending.

    In production, step 2 is replaced by a pre-built vector database index.

    Args:
        query:     The search query text.
        documents: List of document strings to search.
        top_k:     Number of top results to return.

    Returns:
        List of dicts: [{"rank": int, "score": float, "text": str}, ...]
    """
    logger.info("semantic_search: query=%r, doc_count=%d, top_k=%d", query[:50], len(documents), top_k)

    if not documents:
        logger.warning("semantic_search called with empty documents list")
        return []

    # Embed query and all documents in two API calls
    query_vector = get_embedding(query)
    doc_vectors = batch_get_embeddings(documents)

    # Score each document
    scored = []
    for i, (doc, vec) in enumerate(zip(documents, doc_vectors)):
        score = cosine_similarity(query_vector, vec)
        scored.append({"rank": 0, "score": score, "text": doc, "_idx": i})
        logger.debug("Document %d score: %.4f — %r", i, score, doc[:40])

    # Sort by score descending and assign ranks
    scored.sort(key=lambda x: x["score"], reverse=True)
    results = []
    for rank, item in enumerate(scored[:top_k], start=1):
        item["rank"] = rank
        del item["_idx"]
        results.append(item)

    logger.info("Top result: score=%.4f text=%r", results[0]["score"], results[0]["text"][:60])
    return results


def demonstrate_semantic_clustering(
    client_available: bool = True,
) -> Optional[dict]:
    """
    Show that semantically related sentences have higher similarity than unrelated ones.

    This demonstrates the core insight behind embeddings: meaning is geometry.
    """
    logger.info("demonstrate_semantic_clustering: starting")

    sentence_groups = {
        "Travel": [
            "I love visiting Paris in the spring.",
            "The Eiffel Tower is a beautiful landmark.",
            "French cuisine is world-renowned.",
        ],
        "Technology": [
            "Python is a versatile programming language.",
            "Machine learning requires large datasets.",
            "Neural networks are inspired by the brain.",
        ],
    }

    if not client_available:
        logger.info("Skipping clustering demo — no API key")
        return None

    try:
        all_sentences = [s for group in sentence_groups.values() for s in group]
        vectors = batch_get_embeddings(all_sentences)

        print("\n=== Semantic Similarity Matrix ===")
        print(f"{'':40}", end="")
        for i in range(len(all_sentences)):
            print(f" [{i}]", end="")
        print()

        for i, (s_a, v_a) in enumerate(zip(all_sentences, vectors)):
            print(f"[{i}] {s_a[:38]:38}", end="")
            for v_b in vectors:
                sim = cosine_similarity(v_a, v_b)
                print(f" {sim:.2f}", end="")
            print()

        logger.info("Clustering demo complete")
        return {"sentences": all_sentences, "vector_count": len(vectors)}

    except Exception as exc:
        logger.error("Clustering demo failed: %s", exc)
        return None


def main() -> None:
    """Run all embedding demonstrations."""
    logger.info("=== Embeddings Demo Starting ===")

    api_key_available = bool(os.environ.get("OPENAI_API_KEY"))

    if not api_key_available:
        print("\n[!] OPENAI_API_KEY not set — showing concept explanations only.\n")
        print("Embeddings map text to float vectors. Example (simplified 3D):")
        print("  'king'   → [0.8,  0.2, 0.9]")
        print("  'queen'  → [0.8,  0.8, 0.9]  ← similar to king")
        print("  'apple'  → [0.1,  0.1, 0.2]  ← dissimilar")
        print()

        # Demonstrate cosine similarity with hard-coded example vectors
        print("=== Cosine Similarity Examples (no API needed) ===")
        identical = cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        orthogonal = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        partial = cosine_similarity([0.9, 0.1], [0.8, 0.2])
        print(f"  Identical vectors:  {identical:.3f}  (expect 1.000)")
        print(f"  Orthogonal vectors: {orthogonal:.3f}  (expect 0.000)")
        print(f"  Partial overlap:    {partial:.3f}  (expect ~0.98)")
        return

    # ── Full demo when API key is present ─────────────────────────────────────
    print("\n=== 1. Single Embedding ===")
    vec = get_embedding("Artificial intelligence is transforming industries.")
    print(f"  Dimensions: {len(vec)}")
    print(f"  First 5 values: {[round(x, 4) for x in vec[:5]]}")

    print("\n=== 2. Similarity Comparison ===")
    pairs = [
        ("Paris is the capital of France.", "France's capital city is Paris."),
        ("I love pizza.", "Machine learning is fascinating."),
        ("The cat sat on the mat.", "A feline rested on a rug."),
    ]
    for text_a, text_b in pairs:
        va = get_embedding(text_a)
        vb = get_embedding(text_b)
        sim = cosine_similarity(va, vb)
        print(f"  {sim:.3f}  |  {text_a[:40]!r}  vs  {text_b[:40]!r}")

    print("\n=== 3. Semantic Search ===")
    documents = [
        "The Eiffel Tower is located in Paris, France.",
        "Python was created by Guido van Rossum.",
        "Photosynthesis converts sunlight into glucose.",
        "Paris is one of the world's most visited cities.",
        "Neural networks are inspired by the human brain.",
        "France is known for its wine and cuisine.",
    ]

    query = "What can you do in France?"
    print(f"  Query: {query!r}\n")
    results = semantic_search(query, documents, top_k=3)
    for r in results:
        print(f"  [{r['rank']}] score={r['score']:.3f}  {r['text']}")

    print("\n=== 4. Semantic Clustering ===")
    demonstrate_semantic_clustering(api_key_available)

    logger.info("=== Embeddings Demo Complete ===")


if __name__ == "__main__":
    main()
