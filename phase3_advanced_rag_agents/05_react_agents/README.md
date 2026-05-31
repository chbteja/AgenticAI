# 05 — ReAct Agents

## Overview

A ReAct agent (Reason + Act) can use external tools to answer questions it can't answer
from its training data alone. The agent loop:

```
Thought:  I need to find the current weather in Paris.
Action:   weather_tool(city="Paris")
Observation: It's 18°C and sunny.
Thought:  I have the information. I can answer now.
Final Answer: The current weather in Paris is 18°C and sunny.
```

This module builds a ReAct agent with 3 custom tools: a calculator, a knowledge base
search, and a text summariser.

## Learning Objectives

- Understand the ReAct loop: Thought → Action → Observation → repeat
- Build custom LangChain tools with `@tool` decorator
- Create an agent executor with tool binding
- Handle tool errors gracefully
- Observe the agent's reasoning trace

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.3 | Agent framework, tool decorator |
| `langchain-anthropic` | ≥0.3 | LLM backend |
| `python-dotenv` | ≥1.0 | API key loading |

## How to Run

```bash
python react_agent.py
pytest tests/ -v
```

## Key Concept

The `@tool` decorator is the simplest way to create a LangChain tool. The docstring
becomes the tool's description — write it carefully because the LLM reads it to decide
when to use the tool.

**Note**: Phase 6 (LangGraph) replaces `AgentExecutor` with an explicit state graph
that is more debuggable and production-ready.

## Exercises

1. Add a `web_search` tool (mock it for tests).
2. Add a maximum iteration count and observe what happens when the agent loops.
3. Modify the system prompt to change the agent's reasoning style.
