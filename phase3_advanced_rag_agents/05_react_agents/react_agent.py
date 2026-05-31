"""
react_agent.py — ReAct Agents with LangChain Tools

Purpose:
    Build a ReAct (Reason + Act) agent that can use custom tools to answer
    questions requiring external information or computation.

Learning Objectives:
    1. Build custom tools with the @tool decorator.
    2. Create a LangChain agent that binds tools to an LLM.
    3. Trace the Thought → Action → Observation loop.
    4. Handle tool errors gracefully.
    5. Understand why LangGraph replaces AgentExecutor in production.

Tools implemented:
    - calculator: Evaluate arithmetic expressions safely
    - knowledge_search: Search a small in-memory knowledge base
    - summarise_text: Summarise a long text using the LLM itself

Tech Stack: langchain, langchain-anthropic, python-dotenv
"""

import ast
import logging
import operator
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# ── Small in-memory knowledge base for the search tool ────────────────────────
KNOWLEDGE_BASE: dict[str, str] = {
    "python": "Python is a high-level interpreted language created by Guido van Rossum in 1991. "
              "It is widely used in data science, web development, and automation.",
    "langchain": "LangChain is a framework for building LLM-powered applications. "
                 "It provides tools for chains, agents, retrievers, and memory.",
    "rag": "RAG (Retrieval-Augmented Generation) is a technique that retrieves relevant "
           "documents and provides them as context to an LLM to improve answer accuracy.",
    "transformer": "The Transformer architecture, introduced in 'Attention is All You Need' (2017), "
                   "is the foundation of modern LLMs like GPT and Claude.",
    "vector database": "A vector database stores embeddings (high-dimensional float vectors) "
                       "and supports fast similarity search. Examples: Chroma, Pinecone, Weaviate.",
}


# ── Tool definitions ───────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a safe arithmetic expression and return the result.

    Use this tool for any math calculations: addition, subtraction, multiplication,
    division, powers, and parentheses.

    Examples of valid expressions:
        "2 + 2"
        "(10 * 5) / 2"
        "2 ** 10"
        "1000 / 3"

    Args:
        expression: A mathematical expression as a string.

    Returns:
        The numeric result as a string, or an error message.
    """
    logger.info("calculator: expression=%r", expression)

    # Security: only allow numbers and safe math operators — no builtins, no exec
    allowed_pattern = re.compile(r"^[\d\s\+\-\*\/\(\)\.\^]+$")
    if not allowed_pattern.match(expression.replace("**", "^")):
        logger.warning("calculator: rejected unsafe expression: %r", expression)
        return f"Error: Expression contains invalid characters: {expression!r}"

    try:
        # Use ast.literal_eval + operator mapping instead of eval() for security
        # For educational purposes we use a restricted eval with no builtins
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        logger.info("calculator result: %s = %s", expression, result)
        return str(result)
    except Exception as exc:
        logger.error("calculator error: %s", exc)
        return f"Error calculating '{expression}': {exc}"


@tool
def knowledge_search(query: str) -> str:
    """
    Search a knowledge base for information about AI, Python, and related topics.

    Use this tool when you need factual information about:
    - Python programming language
    - LangChain framework
    - RAG (Retrieval-Augmented Generation)
    - Transformer architecture
    - Vector databases

    Args:
        query: A search query (a topic name or question keyword).

    Returns:
        Relevant knowledge base entry, or a message if no match is found.
    """
    logger.info("knowledge_search: query=%r", query)

    query_lower = query.lower().strip()

    # Exact match first
    if query_lower in KNOWLEDGE_BASE:
        result = KNOWLEDGE_BASE[query_lower]
        logger.info("knowledge_search: exact match found")
        return result

    # Partial match — check if the query is a substring of any key
    for key, value in KNOWLEDGE_BASE.items():
        if query_lower in key or key in query_lower:
            logger.info("knowledge_search: partial match on key=%r", key)
            return value

    logger.info("knowledge_search: no match found for %r", query)
    return f"No information found for '{query}'. Available topics: {', '.join(KNOWLEDGE_BASE.keys())}"


@tool
def summarise_text(text: str) -> str:
    """
    Summarise a long piece of text into 2-3 key bullet points.

    Use this tool when you have a long passage and need to extract the main points.
    The text should be at least 100 characters for summarisation to be useful.

    Args:
        text: The text to summarise.

    Returns:
        A 2-3 bullet point summary.
    """
    logger.info("summarise_text: text_len=%d", len(text))

    if len(text) < 50:
        return f"Text too short to summarise: {text!r}"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Cannot summarise: ANTHROPIC_API_KEY not set"

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Summarise the following text in 2-3 bullet points:\n\n{text[:2000]}"
        }],
    )
    summary = response.content[0].text
    logger.info("summarise_text: summary_len=%d", len(summary))
    return summary


# ── Agent factory ──────────────────────────────────────────────────────────────

def build_react_agent(max_iterations: int = 10) -> AgentExecutor:
    """
    Build a ReAct agent with all three tools bound.

    The agent uses tool_calling mode (Claude's native function calling),
    which is more reliable than the older ReAct text-based format.

    Args:
        max_iterations: Maximum number of reasoning steps before stopping.

    Returns:
        AgentExecutor ready to receive queries.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    logger.info("build_react_agent: max_iterations=%d", max_iterations)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        api_key=api_key,
        model=DEFAULT_MODEL,
        max_tokens=1024,
    )

    tools = [calculator, knowledge_search, summarise_text]

    # System prompt — clear instructions help the agent use tools appropriately
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant with access to tools. "
         "Use tools to get information or perform calculations when needed. "
         "Always reason about which tool to use before calling it. "
         "If a tool returns an error, try a different approach."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Shows the reasoning trace — essential for learning
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    logger.info("ReAct agent built with %d tools", len(tools))
    return executor


def run_agent(question: str, executor: AgentExecutor) -> dict:
    """
    Run the agent on a question and return the result with reasoning trace.

    Args:
        question: The question to answer.
        executor: The AgentExecutor to use.

    Returns:
        Dict with keys: question, answer, steps (list of tool calls).
    """
    logger.info("run_agent: question=%r", question[:80])

    result = executor.invoke({"input": question})

    answer = result["output"]
    steps = [
        {
            "tool": step[0].tool,
            "input": step[0].tool_input,
            "output": str(step[1])[:200],
        }
        for step in result.get("intermediate_steps", [])
    ]

    logger.info("run_agent: answer_len=%d, steps=%d", len(answer), len(steps))
    return {"question": question, "answer": answer, "steps": steps}


# ── Demo ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== ReAct Agent Demo Starting ===")

    agent = build_react_agent()

    questions = [
        "What is 17 * 24 + 156?",
        "What is RAG in machine learning?",
        "What is 2 to the power of 8, and what is LangChain?",
        "Search for information about vector databases and summarise what you find.",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        result = run_agent(question, agent)
        print(f"\nAnswer: {result['answer']}")
        if result["steps"]:
            print(f"\nTools used:")
            for step in result["steps"]:
                print(f"  - {step['tool']}({step['input']!r}) → {step['output'][:80]}")

    logger.info("=== ReAct Agent Demo Complete ===")


if __name__ == "__main__":
    main()
