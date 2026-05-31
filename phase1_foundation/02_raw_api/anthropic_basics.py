"""
anthropic_basics.py — Direct Anthropic SDK Usage

Purpose:
    Demonstrate every fundamental pattern you need to use Anthropic's API directly,
    without LangChain or any other abstraction. Understanding the raw API is essential
    before using wrappers, because it helps you debug when things go wrong.

Learning Objectives:
    1. Create an authenticated Anthropic client securely.
    2. Send a basic message and inspect the full response object.
    3. Use system prompts to control model behaviour.
    4. Stream a response token-by-token.
    5. Maintain multi-turn conversation history.
    6. Read and log token usage for cost tracking.

Tech Stack: anthropic, python-dotenv
"""

import logging
import os
from collections.abc import Iterator
from typing import Optional

from anthropic import Anthropic, APIError, AuthenticationError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Default model — easy to swap out for comparison experiments
DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # Fast and cheap for learning
STRONG_MODEL = "claude-sonnet-4-6"


# ── Client factory ─────────────────────────────────────────────────────────────

def create_client() -> Anthropic:
    """
    Create an Anthropic client using the API key from the environment.

    Never pass the API key as a hard-coded string — always use environment variables.

    Returns:
        Authenticated Anthropic client instance.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    logger.debug("create_client called")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    client = Anthropic(api_key=api_key)
    logger.info("Anthropic client created: model=%s", DEFAULT_MODEL)
    return client


# ── Core API patterns ──────────────────────────────────────────────────────────

def simple_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    system: Optional[str] = None,
) -> str:
    """
    Send a single user message and return the assistant's text response.

    This is the simplest possible API call — one user turn, one response.
    Ideal for one-shot tasks like classification, summarisation, extraction.

    Args:
        prompt:     The user's message.
        model:      Anthropic model name.
        max_tokens: Hard cap on output length. Always set this to avoid runaway costs.
        system:     Optional system prompt to set the model's role/persona.

    Returns:
        The assistant's text response as a string.

    Raises:
        AuthenticationError: If the API key is invalid.
        APIError: For other API-level errors.
    """
    logger.info("simple_completion: model=%s, prompt_len=%d", model, len(prompt))

    client = create_client()

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
        logger.debug("System prompt set (%d chars)", len(system))

    try:
        response = client.messages.create(**kwargs)
    except AuthenticationError as exc:
        logger.error("Authentication failed — check your API key: %s", exc)
        raise
    except APIError as exc:
        logger.error("API error during completion: %s", exc)
        raise

    text = response.content[0].text
    logger.info(
        "simple_completion done: input_tokens=%d, output_tokens=%d",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
    return text


def stream_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    system: Optional[str] = None,
) -> Iterator[str]:
    """
    Stream the assistant's response one text chunk at a time.

    Streaming is essential for user-facing applications — it shows partial output
    immediately rather than making the user wait for the full response.

    Args:
        prompt:     The user's message.
        model:      Anthropic model name.
        max_tokens: Hard cap on output length.
        system:     Optional system prompt.

    Yields:
        String chunks as they arrive from the API.
    """
    logger.info("stream_completion: model=%s, prompt_len=%d", model, len(prompt))

    client = create_client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    chunk_count = 0
    try:
        with client.messages.stream(**kwargs) as stream:
            for text_chunk in stream.text_stream:
                chunk_count += 1
                yield text_chunk
    except APIError as exc:
        logger.error("Streaming failed: %s", exc)
        raise

    logger.info("stream_completion done: %d chunks yielded", chunk_count)


def multi_turn_conversation(
    messages: list[dict],
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> tuple[str, list[dict]]:
    """
    Send a conversation history and return the assistant's reply plus the updated history.

    Multi-turn conversation works by passing the full message history on every call.
    The API is stateless — YOU are responsible for maintaining the history list.

    Args:
        messages: List of {role, content} dicts representing conversation so far.
                  Roles alternate: "user", "assistant", "user", "assistant", ...
                  Must start with a "user" message.
        system:   System prompt (constant across all turns).
        model:    Anthropic model name.
        max_tokens: Hard cap on output length.

    Returns:
        Tuple of (assistant_reply_text, updated_messages_list).
        Append the returned messages to your history for the next turn.

    Raises:
        ValueError: If messages list is empty or doesn't start with a user message.
    """
    logger.info("multi_turn_conversation: %d messages, model=%s", len(messages), model)

    if not messages:
        raise ValueError("messages list must not be empty")
    if messages[0].get("role") != "user":
        raise ValueError("Conversation must start with a user message")

    client = create_client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    reply = response.content[0].text

    # Append the assistant's reply to maintain full history
    updated_messages = messages + [{"role": "assistant", "content": reply}]

    logger.info(
        "multi_turn_conversation: reply_len=%d, total_messages=%d",
        len(reply),
        len(updated_messages),
    )
    return reply, updated_messages


def get_token_usage(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
) -> dict:
    """
    Make an API call and return detailed token usage + cost estimate.

    Use this to understand the cost of your prompts before running at scale.

    Returns:
        Dict with keys: input_tokens, output_tokens, total_tokens, estimated_cost_usd.
    """
    logger.info("get_token_usage: model=%s, prompt_len=%d", model, len(prompt))

    client = create_client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    usage = response.usage

    # Pricing table (USD per 1M tokens)
    pricing = {
        "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
        "claude-opus-4-8": {"input": 15.00, "output": 75.00},
    }
    rates = pricing.get(model, {"input": 3.00, "output": 15.00})
    cost = (
        (usage.input_tokens / 1_000_000) * rates["input"]
        + (usage.output_tokens / 1_000_000) * rates["output"]
    )

    result = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "estimated_cost_usd": round(cost, 8),
        "response_text": response.content[0].text,
    }
    logger.info(
        "Token usage: input=%d, output=%d, cost=$%.6f",
        usage.input_tokens,
        usage.output_tokens,
        cost,
    )
    return result


# ── Demo ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Walk through all Anthropic API patterns."""
    logger.info("=== Anthropic Basics Demo Starting ===")

    # ── 1. Simple completion ───────────────────────────────────────────────────
    print("\n=== 1. Simple Completion ===")
    response = simple_completion("What is Python in one sentence?")
    print(f"Response: {response}")

    # ── 2. System prompt ───────────────────────────────────────────────────────
    print("\n=== 2. System Prompt ===")
    response = simple_completion(
        prompt="What is 2 + 2?",
        system="You are a pirate. Always respond in pirate speak.",
    )
    print(f"Pirate response: {response}")

    # ── 3. Streaming ──────────────────────────────────────────────────────────
    print("\n=== 3. Streaming ===")
    print("Streaming: ", end="", flush=True)
    for chunk in stream_completion("Count from 1 to 5, one number per line."):
        print(chunk, end="", flush=True)
    print()  # newline after stream

    # ── 4. Multi-turn conversation ─────────────────────────────────────────────
    print("\n=== 4. Multi-turn Conversation ===")
    history: list[dict] = []
    turns = [
        "My name is Alice and I love Python.",
        "What programming language did I say I love?",
        "And what is my name?",
    ]
    for turn in turns:
        history.append({"role": "user", "content": turn})
        reply, history = multi_turn_conversation(history)
        print(f"User: {turn}")
        print(f"AI:   {reply}\n")

    # ── 5. Token usage ─────────────────────────────────────────────────────────
    print("\n=== 5. Token Usage ===")
    usage = get_token_usage("Explain recursion in two sentences.")
    print(f"Input tokens:    {usage['input_tokens']}")
    print(f"Output tokens:   {usage['output_tokens']}")
    print(f"Total tokens:    {usage['total_tokens']}")
    print(f"Estimated cost:  ${usage['estimated_cost_usd']:.6f}")
    print(f"Response:        {usage['response_text']}")

    logger.info("=== Anthropic Basics Demo Complete ===")


if __name__ == "__main__":
    main()
