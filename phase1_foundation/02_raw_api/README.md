# 02 — Raw API Calls (Anthropic + OpenAI)

## Overview

Before reaching for LangChain, you need to understand what it is wrapping.
This module makes direct API calls to Anthropic and OpenAI using their official Python SDKs.
You will see the full request/response structure, handle streaming, maintain conversation history,
and understand token usage — all without any framework magic.

## Learning Objectives

- Set up Anthropic and OpenAI clients correctly and securely
- Send a basic completion request and inspect the response object
- Stream responses token-by-token for real-time output
- Maintain multi-turn conversation history manually
- Read token usage from response objects and understand billing

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `anthropic` | ≥0.40 | Official Anthropic Python SDK |
| `openai` | ≥1.50 | Official OpenAI Python SDK |
| `python-dotenv` | ≥1.0 | Load API keys from `.env` |

## Setup

```bash
cp ../../.env.example ../../.env
# Fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

## How to Run

```bash
# Anthropic examples
python anthropic_basics.py

# OpenAI examples
python openai_basics.py

# Tests (mocked — no API key needed)
pytest tests/ -v
```

## Key Concepts

- **Messages API**: Anthropic and OpenAI both use a `messages` array format: `[{"role": "user", "content": "..."}]`
- **System prompt**: Sets the AI's persona and constraints — passed separately in Anthropic, as the first message in OpenAI
- **Streaming**: The model sends tokens as they are generated; your code receives an iterator of chunks
- **Token usage**: Every response includes `input_tokens` + `output_tokens` — track these for cost monitoring
- **Multi-turn**: Conversation history is just an ever-growing list of `{role, content}` dicts

## Expected Output

```
=== Simple Completion ===
Response: Python is a high-level, interpreted programming language...

=== Streaming ===
Streaming: The capital of France is Paris...

=== Multi-turn Conversation ===
Turn 1 — User: My name is Alice.
Turn 1 — AI:   Hello Alice! Nice to meet you...
Turn 2 — User: What's my name?
Turn 2 — AI:   Your name is Alice...

=== Token Usage ===
Input tokens:  42
Output tokens: 87
Estimated cost: $0.00018
```

## Exercises

1. Add a `temperature` parameter and observe how higher values produce more varied responses.
2. Implement a simple chatbot loop using `multi_turn_conversation`.
3. Compare the same prompt sent to `claude-sonnet-4-6` vs `claude-haiku-4-5-20251001` — observe quality/speed/cost trade-offs.
