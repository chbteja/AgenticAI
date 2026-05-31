"""
langgraph_agent.py — Production ReAct Agent with LangGraph

Purpose:
    Rebuild the ReAct agent from module 05 using LangGraph's StateGraph.
    LangGraph makes control flow explicit — you can see, modify, and debug
    every step of the agent loop rather than relying on AgentExecutor's black box.

Learning Objectives:
    1. Define agent state as a TypedDict.
    2. Build a StateGraph with agent_node, tools_node, and conditional edges.
    3. Understand how messages accumulate in state across tool calls.
    4. Add persistence with MemorySaver for multi-turn conversations.
    5. Inspect the full execution trace.

Architecture:
    START → agent_node → (conditional) → tools_node → agent_node → ... → END

Tech Stack: langgraph, langchain-anthropic, python-dotenv
"""

import logging
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# ── State definition ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The state object passed between every node in the graph.

    The `add_messages` annotation tells LangGraph to APPEND new messages
    rather than overwrite the list — this is how the conversation history builds up.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int  # How many LLM calls have been made (for loop control)


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """
    Evaluate an arithmetic expression (numbers and +, -, *, /, **, parentheses only).

    Args:
        expression: e.g. "2 ** 8" or "(100 + 50) / 3"

    Returns:
        The numeric result as a string.
    """
    logger.info("calculate: %r", expression)
    import re
    if not re.match(r"^[\d\s\+\-\*\/\(\)\.\^]+$", expression.replace("**", "^")):
        return f"Error: unsafe expression {expression!r}"
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@tool
def lookup_fact(topic: str) -> str:
    """
    Look up a fact about a technology topic.

    Topics available: python, langchain, rag, langgraph, transformer, llm.

    Args:
        topic: The topic to look up (case-insensitive).

    Returns:
        A fact about the topic, or a not-found message.
    """
    logger.info("lookup_fact: %r", topic)
    facts = {
        "python": "Python was created by Guido van Rossum and released in 1991. It is dynamically typed.",
        "langchain": "LangChain is a framework for composing LLM applications with chains, agents, and retrievers.",
        "rag": "RAG combines retrieval (finding relevant docs) with generation (LLM answers) to reduce hallucination.",
        "langgraph": "LangGraph extends LangChain with explicit state machines, enabling cyclical workflows and persistence.",
        "transformer": "The Transformer (2017) introduced self-attention, replacing RNNs for sequence modelling.",
        "llm": "Large Language Models are trained on massive text corpora using self-supervised next-token prediction.",
    }
    return facts.get(topic.lower(), f"No fact found for '{topic}'. Try: {', '.join(facts.keys())}")


# ── LLM factory ────────────────────────────────────────────────────────────────

def _get_llm_with_tools():
    """Create an LLM with tools bound, ready for tool_calling."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(api_key=api_key, model=DEFAULT_MODEL, max_tokens=1024)
    tools = [calculate, lookup_fact]
    return llm.bind_tools(tools), tools


# ── Graph nodes ────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """
    The main reasoning node. Sends messages to the LLM and receives a response.

    The LLM response may contain:
    - A `tool_calls` list → we should call those tools (go to tools_node)
    - Just text → we're done (go to END)

    Args:
        state: Current agent state with message history.

    Returns:
        Updated state dict with the new AI message appended.
    """
    logger.info("agent_node: step=%d, messages=%d", state["step_count"], len(state["messages"]))

    llm_with_tools, _ = _get_llm_with_tools()
    response = llm_with_tools.invoke(state["messages"])

    logger.info(
        "agent_node response: has_tool_calls=%s",
        bool(getattr(response, "tool_calls", None)),
    )
    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
    }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge: examine the last message to decide what to do next.

    If the last message has tool_calls → route to tools_node.
    Otherwise → route to END.

    This function is the heart of LangGraph's power: control flow as code.

    Args:
        state: Current agent state.

    Returns:
        "tools" or "end" — the name of the next node.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        logger.info("should_continue: routing to tools (%d calls)", len(tool_calls))
        return "tools"

    # Safety guard: stop if we've taken too many steps (prevent infinite loops)
    if state["step_count"] >= 10:
        logger.warning("should_continue: max steps reached — forcing END")
        return "end"

    logger.info("should_continue: no tool calls — routing to END")
    return "end"


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Graph structure:
        START → agent_node
        agent_node → (conditional) → tools OR end
        tools → agent_node  (loop)

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    logger.info("build_graph: constructing")

    _, tools = _get_llm_with_tools()
    tools_node = ToolNode(tools)

    # Create the state graph with our AgentState schema
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    # Entry point
    graph.add_edge(START, "agent")

    # Conditional edge: after agent runs, decide whether to call tools or stop
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # If tool calls present → go to tools
            "end": END,        # If no tool calls → end
        },
    )

    # After tools run → always go back to agent for another reasoning step
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    logger.info("Graph compiled successfully")
    return compiled


def build_graph_with_memory():
    """
    Build the graph with MemorySaver for persistent conversation history.

    With persistence, the graph can resume a conversation thread across invocations.
    Use thread_id to identify conversations.

    Returns:
        Compiled graph with checkpointing enabled.
    """
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("build_graph_with_memory: constructing")

    _, tools = _get_llm_with_tools()
    tools_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    logger.info("Graph with memory compiled")
    return compiled


def run_graph(question: str, graph) -> dict:
    """
    Run the graph on a question and return the final answer with trace.

    Args:
        question: The user's question.
        graph:    Compiled LangGraph graph.

    Returns:
        Dict with keys: question, answer, steps (tool calls made).
    """
    logger.info("run_graph: question=%r", question[:80])

    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "step_count": 0,
    }

    final_state = graph.invoke(initial_state)
    final_message = final_state["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    # Extract tool calls from message history
    steps = []
    for msg in final_state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({"tool": tc["name"], "input": tc["args"]})

    logger.info(
        "run_graph: done, steps=%d, answer_len=%d",
        len(steps), len(str(answer)),
    )
    return {"question": question, "answer": answer, "steps": steps}


# ── Demo ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== LangGraph Agent Demo Starting ===")

    print("\n=== Building LangGraph Agent ===")
    graph = build_graph()
    print("Graph compiled successfully")

    questions = [
        "What is 256 divided by 16?",
        "What is LangGraph and what is RAG?",
        "Calculate 3 ** 4 and then look up what LLMs are.",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        result = run_graph(question, graph)
        print(f"Answer:   {result['answer']}")
        if result["steps"]:
            print("Tools used:")
            for step in result["steps"]:
                print(f"  - {step['tool']}({step['input']})")

    # ── Multi-turn with memory ─────────────────────────────────────────────────
    print("\n=== Multi-turn Conversation with Memory ===")
    memory_graph = build_graph_with_memory()
    thread = {"configurable": {"thread_id": "demo-thread-1"}}

    turns = [
        "My name is Alice and I'm learning about AI.",
        "What is RAG?",
        "What was my name again?",  # Tests memory
    ]

    for turn in turns:
        print(f"\nUser: {turn}")
        result = memory_graph.invoke(
            {"messages": [HumanMessage(content=turn)], "step_count": 0},
            config=thread,
        )
        answer = result["messages"][-1].content
        print(f"AI:   {answer}")

    logger.info("=== LangGraph Agent Demo Complete ===")


if __name__ == "__main__":
    main()
