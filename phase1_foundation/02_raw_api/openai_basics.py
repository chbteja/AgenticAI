"""
openai_basics.py — Direct OpenAI SDK Usage

Purpose:
    Mirror of anthropic_basics.py using the OpenAI SDK.
    Demonstrates the same patterns (simple completion, streaming, multi-turn,
    structured JSON output) so students can compare both SDKs side-by-side.

Learning Objectives:
    1. Create an OpenAI client securely.
    2. Send chat completion requests.
    3. Stream responses chunk-by-chunk.
    4. Maintain multi-turn history manually.
    5. Request structured JSON output using response_format.

Tech Stack: openai, python-dotenv
"""

import json
import logging
import os
from collections.abc import Iterator
from typing import Optional

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, OpenAIError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"  # Fast and cheap for learning experiments


# ── Client factory ─────────────────────────────────────────────────────────────

def create_client() -> OpenAI:
    """
    Create an OpenAI client using the API key from the environment.

    Returns:
        Authenticated OpenAI client.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    logger.debug("create_client called")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client created: model=%s", DEFAULT_MODEL)
    return client


# ── Core API patterns ──────────────────────────────────────────────────────────

def chat_completion(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Send a list of messages and return the assistant's text reply.

    OpenAI uses the messages array for everything — system prompts are
    included as the first message with role "system".

    Args:
        messages:    List of {role, content} dicts. Role can be: system, user, assistant.
        model:       OpenAI model name.
        max_tokens:  Hard cap on output length.
        temperature: Sampling temperature. 0.0 = deterministic, 1.0 = creative.

    Returns:
        The assistant's text response.

    Raises:
        AuthenticationError: If the API key is invalid.
        OpenAIError: For other API errors.
    """
    logger.info("chat_completion: model=%s, messages=%d", model, len(messages))

    if not messages:
        raise ValueError("messages list must not be empty")

    client = create_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except AuthenticationError as exc:
        logger.error("Authentication failed — check OPENAI_API_KEY: %s", exc)
        raise
    except OpenAIError as exc:
        logger.error("OpenAI API error: %s", exc)
        raise

    text = response.choices[0].message.content
    logger.info(
        "chat_completion done: prompt_tokens=%d, completion_tokens=%d",
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )
    return text


def stream_chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """
    Stream a chat completion one delta chunk at a time.

    Args:
        messages:   List of {role, content} dicts.
        model:      OpenAI model name.
        max_tokens: Hard cap on output length.

    Yields:
        String delta chunks as they arrive from the API.
    """
    logger.info("stream_chat: model=%s, messages=%d", model, len(messages))

    client = create_client()
    chunk_count = 0

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                chunk_count += 1
                yield delta
    except OpenAIError as exc:
        logger.error("Streaming failed: %s", exc)
        raise

    logger.info("stream_chat done: %d chunks", chunk_count)


def multi_turn_conversation(
    user_message: str,
    history: list[dict],
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> tuple[str, list[dict]]:
    """
    Append a new user message to history, get a reply, and return updated history.

    The system prompt is prepended automatically if provided and not already present.

    Args:
        user_message: The new message from the user.
        history:      Previous conversation turns. May be empty for the first turn.
        system:       System prompt (applied only if history is empty).
        model:        OpenAI model name.

    Returns:
        Tuple of (assistant_reply, updated_history).
    """
    logger.info("multi_turn_conversation: history_len=%d", len(history))

    messages = list(history)  # Don't mutate the caller's list

    # Prepend system message on first turn
    if system and not messages:
        messages.insert(0, {"role": "system", "content": system})

    messages.append({"role": "user", "content": user_message})
    reply = chat_completion(messages, model=model)
    messages.append({"role": "assistant", "content": reply})

    logger.info("multi_turn_conversation: reply_len=%d, total_turns=%d", len(reply), len(messages))
    return reply, messages


def structured_json_output(
    prompt: str,
    schema_description: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Request JSON-formatted output from the model.

    Uses response_format={"type": "json_object"} to guarantee valid JSON.
    The schema_description tells the model what JSON shape to produce.

    Args:
        prompt:             The task or question.
        schema_description: Natural language description of the expected JSON shape.
        model:              OpenAI model name (json_object mode requires gpt-4o or gpt-3.5-turbo-1106+).

    Returns:
        Parsed Python dict from the model's JSON response.

    Raises:
        json.JSONDecodeError: If the model returns invalid JSON (rare with json_object mode).
    """
    logger.info("structured_json_output: model=%s, prompt_len=%d", model, len(prompt))

    system_msg = (
        "You always respond with valid JSON. "
        f"The JSON should match this structure: {schema_description}"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    client = create_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=1024,
    )

    raw = response.choices[0].message.content
    logger.info("structured_json_output: raw_len=%d", len(raw))

    try:
        parsed = json.loads(raw)
        logger.info("structured_json_output: parsed %d top-level keys", len(parsed))
        return parsed
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON response: %s\nRaw: %s", exc, raw[:200])
        raise


# ── Demo ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Walk through all OpenAI API patterns."""
    logger.info("=== OpenAI Basics Demo Starting ===")

    # ── 1. Simple chat completion ──────────────────────────────────────────────
    print("\n=== 1. Chat Completion ===")
    reply = chat_completion([{"role": "user", "content": "What is machine learning in one sentence?"}])
    print(f"Response: {reply}")

    # ── 2. System prompt ───────────────────────────────────────────────────────
    print("\n=== 2. System Prompt ===")
    reply = chat_completion(
        messages=[
            {"role": "system", "content": "You are a Shakespearean poet. Respond only in iambic pentameter."},
            {"role": "user", "content": "What is Python?"},
        ]
    )
    print(f"Shakespearean response: {reply}")

    # ── 3. Streaming ──────────────────────────────────────────────────────────
    print("\n=== 3. Streaming ===")
    print("Streaming: ", end="", flush=True)
    for chunk in stream_chat([{"role": "user", "content": "List 3 planets in the solar system."}]):
        print(chunk, end="", flush=True)
    print()

    # ── 4. Multi-turn conversation ─────────────────────────────────────────────
    print("\n=== 4. Multi-turn Conversation ===")
    history: list[dict] = []
    system_prompt = "You are a helpful coding assistant."
    turns = [
        "My favourite language is Rust.",
        "What's my favourite language?",
        "What is that language best known for?",
    ]
    for turn in turns:
        reply, history = multi_turn_conversation(turn, history, system_prompt)
        print(f"User: {turn}")
        print(f"AI:   {reply}\n")

    # ── 5. Structured JSON output ──────────────────────────────────────────────
    print("\n=== 5. Structured JSON Output ===")
    result = structured_json_output(
        prompt="Extract the name, age, and city from: 'John is 30 years old and lives in London.'",
        schema_description='{"name": "string", "age": "integer", "city": "string"}',
    )
    print(f"Parsed JSON: {json.dumps(result, indent=2)}")

    logger.info("=== OpenAI Basics Demo Complete ===")


if __name__ == "__main__":
    main()
