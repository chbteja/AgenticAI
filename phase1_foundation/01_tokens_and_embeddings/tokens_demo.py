"""
tokens_demo.py — Understanding LLM Tokenization

Purpose:
    Demonstrate how large language models convert text into tokens (integers),
    and how to count tokens before making API calls for cost estimation.

Learning Objectives:
    1. Understand the tiktoken tokenizer used by OpenAI models.
    2. Understand how Anthropic counts tokens via its API.
    3. Build a cost estimator for production workloads.
    4. Observe how different text types (code, emoji, foreign text) tokenize differently.

Tech Stack: tiktoken, anthropic, python-dotenv
"""

import logging
import os
from typing import Optional

import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env (never hardcode API keys)
load_dotenv()

# Configure structured logging — every module uses this same format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Pricing table (USD per 1M tokens, as of late 2024) ────────────────────────
# Update these when pricing changes — do NOT hardcode in the calculation logic.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
}


def count_tokens_tiktoken(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in *text* using the tiktoken tokenizer.

    tiktoken is fast and works offline — no API call required.
    Use this for OpenAI models; Anthropic's tokenizer differs slightly.

    Args:
        text:  The text to tokenize.
        model: OpenAI model name. Determines which BPE encoding to use.

    Returns:
        Integer token count.

    Raises:
        KeyError: If the model has no known encoding in tiktoken.
    """
    logger.debug("count_tokens_tiktoken called: model=%s, text_len=%d", model, len(text))

    if not text:
        logger.warning("Empty text passed to count_tokens_tiktoken — returning 0")
        return 0

    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to the most common encoding when model is not in tiktoken's registry
        logger.warning("Model %r not in tiktoken registry; falling back to cl100k_base", model)
        encoder = tiktoken.get_encoding("cl100k_base")

    tokens = encoder.encode(text)
    count = len(tokens)
    logger.debug("count_tokens_tiktoken result: %d tokens", count)
    return count


def inspect_token_breakdown(text: str, model: str = "gpt-4o") -> list[tuple[int, str]]:
    """
    Return a list of (token_id, decoded_text_piece) for every token in *text*.

    This is the best way to understand how a tokenizer splits your prompt.
    Useful for debugging unexpected token counts.

    Args:
        text:  The text to inspect.
        model: OpenAI model name.

    Returns:
        List of (int, str) pairs — one per token.
    """
    logger.debug("inspect_token_breakdown called: model=%s", model)

    if not text:
        return []

    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")

    token_ids = encoder.encode(text)
    breakdown = [(tid, encoder.decode([tid])) for tid in token_ids]

    logger.info("Token breakdown: %d tokens for %d characters", len(breakdown), len(text))
    return breakdown


def count_tokens_anthropic(text: str, model: str = "claude-sonnet-4-6") -> Optional[int]:
    """
    Count tokens using Anthropic's official token counting endpoint.

    This makes a real API call and requires ANTHROPIC_API_KEY. It is more accurate
    than tiktoken for Anthropic models because it uses the same tokenizer the model uses.

    Args:
        text:  The text to count tokens for (used as a user message).
        model: Anthropic model name.

    Returns:
        Integer token count, or None if the API key is not configured.
    """
    logger.debug("count_tokens_anthropic called: model=%s", model)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping Anthropic token count")
        return None

    try:
        # Import here so the module works without anthropic if only tiktoken is needed
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        response = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        count = response.input_tokens
        logger.info("Anthropic token count for model %s: %d tokens", model, count)
        return count
    except Exception as exc:
        logger.error("Anthropic token count failed: %s", exc)
        return None


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet-4-6",
) -> dict[str, float]:
    """
    Estimate the USD cost for a given number of input + output tokens.

    Args:
        input_tokens:  Number of prompt / input tokens.
        output_tokens: Number of completion / output tokens.
        model:         Model name (must be in MODEL_PRICING).

    Returns:
        Dict with keys: input_cost, output_cost, total_cost (all in USD).

    Raises:
        ValueError: If the model is not in the pricing table.
    """
    logger.debug("estimate_cost called: model=%s, in=%d, out=%d", model, input_tokens, output_tokens)

    if model not in MODEL_PRICING:
        raise ValueError(
            f"Unknown model {model!r}. Known models: {list(MODEL_PRICING.keys())}"
        )

    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    result = {
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(total_cost, 8),
    }
    logger.info("Cost estimate for %s: $%.6f total", model, total_cost)
    return result


def _print_separator(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main() -> None:
    """
    Interactive demo covering all tokenization concepts.

    Run this directly to see how different types of text tokenize.
    """
    logger.info("=== Tokens Demo Starting ===")

    # ── 1. Basic token counts ──────────────────────────────────────────────────
    _print_separator("1. Token Counts (tiktoken / gpt-4o)")

    examples = [
        ("Simple English sentence", "Hello, world! How are you today?"),
        ("Programming code", "def fibonacci(n: int) -> int: return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
        ("Long uncommon word", "Supercalifragilisticexpialidocious"),
        ("Mathematical expression", "∫₀^∞ e^(-x²) dx = √π/2"),
        ("Japanese text", "日本語のテキストはトークン数が多い"),
        ("Emoji", "🚀🤖🧠💡"),
        ("Numbers", "1234567890 vs one billion two hundred thirty-four million"),
    ]

    for label, text in examples:
        count = count_tokens_tiktoken(text)
        ratio = len(text) / count if count else 0
        print(f"  {label:<30} {count:4d} tokens  ({ratio:.1f} chars/token)")

    # ── 2. Token breakdown for a short phrase ─────────────────────────────────
    _print_separator("2. Token Breakdown — 'tokenization'")

    phrase = "tokenization breaks text into pieces"
    breakdown = inspect_token_breakdown(phrase)
    print(f"  Text: {phrase!r}\n")
    for i, (tid, piece) in enumerate(breakdown):
        print(f"  [{i:2d}] id={tid:6d}  piece={piece!r}")

    # ── 3. Context window limits ───────────────────────────────────────────────
    _print_separator("3. Context Window Reference")

    context_windows = {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "claude-sonnet-4-6": 200_000,
        "claude-haiku-4-5-20251001": 200_000,
        "claude-opus-4-8": 200_000,
    }

    for model, limit in context_windows.items():
        pages_approx = limit // 500  # ~500 tokens per A4 page
        print(f"  {model:<35} {limit:>8,} tokens  (~{pages_approx:,} pages)")

    # ── 4. Cost estimation ────────────────────────────────────────────────────
    _print_separator("4. Cost Estimation — 10,000 input + 500 output tokens")

    for model in MODEL_PRICING:
        costs = estimate_cost(10_000, 500, model)
        print(f"  {model:<35} ${costs['total_cost']:.5f}")

    # ── 5. Anthropic API token count (only if key is available) ───────────────
    _print_separator("5. Anthropic API Token Count")

    test_text = "Explain the concept of attention mechanisms in transformers."
    anthropic_count = count_tokens_anthropic(test_text)
    tiktoken_count = count_tokens_tiktoken(test_text)

    if anthropic_count is not None:
        print(f"  Text: {test_text!r}")
        print(f"  tiktoken count:   {tiktoken_count}")
        print(f"  Anthropic count:  {anthropic_count}")
        print(f"  Difference:       {abs(tiktoken_count - anthropic_count)}")
    else:
        print("  (Skipped — ANTHROPIC_API_KEY not set)")
        print(f"  tiktoken count for reference: {tiktoken_count}")

    logger.info("=== Tokens Demo Complete ===")


if __name__ == "__main__":
    main()
