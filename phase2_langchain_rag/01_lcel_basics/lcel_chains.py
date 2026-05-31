"""
lcel_chains.py — LangChain Expression Language (LCEL) Basics

Purpose:
    Teach LCEL, the modern pipe-based composition system in LangChain.
    Every component is a Runnable; the | operator builds chains that automatically
    support streaming, batching, async, and tracing.

Learning Objectives:
    1. Build a simple PromptTemplate | LLM | OutputParser chain.
    2. Use RunnableParallel to execute multiple chains concurrently.
    3. Use RunnablePassthrough to inject data into a chain without transforming it.
    4. Stream LCEL output token-by-token.
    5. Use .batch() for parallel processing of multiple inputs.

Tech Stack: langchain, langchain-anthropic, python-dotenv
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# ── LLM factory ────────────────────────────────────────────────────────────────

def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0):
    """
    Create a LangChain-compatible Anthropic chat model.

    The returned object implements the Runnable interface, so it can be
    used directly in | chains.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    logger.debug("get_llm: model=%s, temperature=%s", model, temperature)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — see .env.example")

    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=1024,
    )
    logger.info("LLM created: %s", model)
    return llm


# ── Chain builders ─────────────────────────────────────────────────────────────

def build_simple_chain(model: str = DEFAULT_MODEL):
    """
    Build the simplest possible LCEL chain: prompt | llm | parser.

    The chain takes a dict {"topic": str} and returns a string.

    Chain anatomy:
        PromptTemplate — fills {topic} into the template
        ChatAnthropic  — sends the formatted prompt to the API
        StrOutputParser — extracts just the text from the AIMessage response

    Returns:
        A Runnable chain that accepts {"topic": str} and returns str.
    """
    logger.debug("build_simple_chain: building")
    llm = get_llm(model)

    prompt = ChatPromptTemplate.from_template(
        "Answer in one sentence: What is {topic}?"
    )
    parser = StrOutputParser()

    chain = prompt | llm | parser
    logger.info("Simple chain built: PromptTemplate | LLM | StrOutputParser")
    return chain


def build_parallel_chain(model: str = DEFAULT_MODEL):
    """
    Build a parallel chain that runs two tasks on the same input simultaneously.

    RunnableParallel executes both branches concurrently and returns:
        {"summary": str, "translation": str}

    This is useful when you need multiple perspectives on the same content.

    Returns:
        A Runnable that accepts {"text": str} and returns {"summary": str, "translation": str}.
    """
    logger.debug("build_parallel_chain: building")
    llm = get_llm(model)
    parser = StrOutputParser()

    summary_prompt = ChatPromptTemplate.from_template(
        "Summarise this text in one sentence: {text}"
    )
    translate_prompt = ChatPromptTemplate.from_template(
        "Translate this text to French: {text}"
    )

    summary_chain = summary_prompt | llm | parser
    translate_chain = translate_prompt | llm | parser

    parallel_chain = RunnableParallel(
        summary=summary_chain,
        translation=translate_chain,
    )
    logger.info("Parallel chain built: summary + translation")
    return parallel_chain


def build_passthrough_chain(model: str = DEFAULT_MODEL):
    """
    Demonstrate RunnablePassthrough — pass data through unchanged alongside a transformation.

    Output: {"original": str, "improved": str}

    RunnablePassthrough() is the identity function for Runnables.
    Use it when you need to preserve the original input while also transforming it.

    Returns:
        A Runnable that accepts {"text": str} and returns {"original": str, "improved": str}.
    """
    logger.debug("build_passthrough_chain: building")
    llm = get_llm(model)
    parser = StrOutputParser()

    improve_prompt = ChatPromptTemplate.from_template(
        "Improve this text for clarity and conciseness: {text}"
    )
    improve_chain = improve_prompt | llm | parser

    chain = RunnableParallel(
        original=RunnablePassthrough(),
        improved=improve_chain,
    )
    logger.info("Passthrough chain built")
    return chain


def build_sequential_chain(model: str = DEFAULT_MODEL):
    """
    Build a two-step sequential chain: generate a fact, then explain it.

    Demonstrates how to pass the output of one chain as input to another.
    Uses RunnablePassthrough.assign() to add new keys to the context dict.

    Returns:
        A Runnable accepting {"topic": str} and returning {"fact": str, "explanation": str}.
    """
    logger.debug("build_sequential_chain: building")
    llm = get_llm(model)
    parser = StrOutputParser()

    # Step 1: Generate an interesting fact
    fact_prompt = ChatPromptTemplate.from_template(
        "State one surprising fact about {topic} in one sentence."
    )
    fact_chain = fact_prompt | llm | parser

    # Step 2: Explain the fact (receives the fact from step 1)
    explain_prompt = ChatPromptTemplate.from_template(
        "Explain why this is true or interesting: {fact}"
    )
    explain_chain = explain_prompt | llm | parser

    # Chain them: first generate fact, then explain it
    chain = (
        RunnablePassthrough.assign(fact=fact_chain)
        | RunnablePassthrough.assign(explanation=explain_chain)
    )
    logger.info("Sequential chain built: fact → explanation")
    return chain


# ── Demo ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Demonstrate all LCEL patterns with live API calls."""
    logger.info("=== LCEL Basics Demo Starting ===")

    # ── 1. Simple chain ────────────────────────────────────────────────────────
    print("\n=== 1. Simple Chain (prompt | llm | parser) ===")
    chain = build_simple_chain()
    result = chain.invoke({"topic": "the Python programming language"})
    print(f"Result: {result}")

    # ── 2. Parallel chain ──────────────────────────────────────────────────────
    print("\n=== 2. Parallel Chain (summary + translation) ===")
    parallel = build_parallel_chain()
    text = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
    result = parallel.invoke({"text": text})
    print(f"Summary:     {result['summary']}")
    print(f"Translation: {result['translation']}")

    # ── 3. Passthrough chain ───────────────────────────────────────────────────
    print("\n=== 3. Passthrough Chain (preserve + transform) ===")
    passthrough = build_passthrough_chain()
    original_text = "The thing that this does is it makes the code run faster by using a better algorithm."
    result = passthrough.invoke({"text": original_text})
    print(f"Original: {result['original']['text']}")
    print(f"Improved: {result['improved']}")

    # ── 4. Sequential chain ────────────────────────────────────────────────────
    print("\n=== 4. Sequential Chain (fact → explanation) ===")
    sequential = build_sequential_chain()
    result = sequential.invoke({"topic": "black holes"})
    print(f"Fact:        {result['fact']}")
    print(f"Explanation: {result['explanation']}")

    # ── 5. Streaming ──────────────────────────────────────────────────────────
    print("\n=== 5. Streaming ===")
    chain = build_simple_chain()
    print("Streaming: ", end="", flush=True)
    for chunk in chain.stream({"topic": "the speed of light"}):
        print(chunk, end="", flush=True)
    print()

    # ── 6. Batch processing ────────────────────────────────────────────────────
    print("\n=== 6. Batch Processing ===")
    chain = build_simple_chain()
    topics = [
        {"topic": "recursion in programming"},
        {"topic": "the water cycle"},
        {"topic": "supply and demand"},
    ]
    logger.info("Running batch of %d inputs", len(topics))
    results = chain.batch(topics)
    for topic_dict, answer in zip(topics, results):
        print(f"Q: {topic_dict['topic']!r}")
        print(f"A: {answer}\n")

    logger.info("=== LCEL Basics Demo Complete ===")


if __name__ == "__main__":
    main()
