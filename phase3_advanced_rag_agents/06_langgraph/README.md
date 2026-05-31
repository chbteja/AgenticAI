# 06 — LangGraph Agents

## Overview

LangGraph is the right way to build production ReAct agents. Unlike the older `AgentExecutor`,
LangGraph makes the control flow **explicit code** — nodes, edges, and conditional branches —
making it far easier to debug, customise, and extend.

```
            ┌─────────────┐
            │  START      │
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  agent_node │ ◄─── LLM decides: answer or call tool
            └──────┬──────┘
                   │
            ┌──────▼──────────────────────────┐
            │ should_continue (conditional)?  │
            │  if tool_call → tools_node      │
            │  if final_answer → END          │
            └──────┬──────────────────────────┘
                   │
            ┌──────▼──────┐
            │  tools_node │ ─── Execute tool, append result
            └──────┬──────┘
                   │
                 (loop back to agent_node)
```

## Learning Objectives

- Understand LangGraph's StateGraph, nodes, and edges
- Build a graph with conditional edges (tool call vs final answer)
- Use TypedDict for typed state management
- Compare LangGraph vs AgentExecutor for debugging

## How to Run

```bash
python langgraph_agent.py
pytest tests/ -v
```

## Key Concepts

- **StateGraph**: A graph where nodes are functions and edges define transitions
- **State**: A TypedDict passed between nodes — each node reads and writes to state
- **Conditional edge**: A function that inspects state and returns the next node name
- **ToolNode**: A built-in LangGraph node that executes all pending tool calls

## Exercises

1. Add a `max_steps` field to the state and add an early-exit edge when exceeded.
2. Add a `human_in_the_loop` node that pauses and waits for user approval before calling a tool.
3. Persist state between runs using `MemorySaver` checkpointing.
