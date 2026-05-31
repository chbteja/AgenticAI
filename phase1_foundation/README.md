# Phase 1 — Foundation (Weeks 1–3)

> "Understand WHY things work, not just copy code."

This phase builds the mental model that underpins everything else in the curriculum.
You will learn how LLMs see text (tokens), how meaning is encoded (embeddings),
how to call APIs directly without abstractions, and how to engineer prompts systematically.

---

## Modules

| Module | What You Build | Key Skill |
|--------|---------------|-----------|
| `01_tokens_and_embeddings` | Token counter + semantic search | How text becomes numbers |
| `02_raw_api` | Direct LLM calls (Anthropic + OpenAI) | API literacy before LangChain |
| `03_prompt_engineering` | Prompt comparison harness | Reliable LLM output patterns |

## Learning Path

1. Start with `01_tokens_and_embeddings/tokens_demo.py` — run it and observe how different text types tokenize differently.
2. Explore `embeddings_demo.py` — try adding your own sentences to the similarity search.
3. Move to `02_raw_api/` — send your first raw API calls in both SDKs.
4. Finish with `03_prompt_engineering/` — run the technique comparison and notice output quality differences.

## Key Insight

Every LangChain abstraction you learn in Phase 2 is just a wrapper around what you do here manually.
Understanding the raw layer makes debugging dramatically easier.
