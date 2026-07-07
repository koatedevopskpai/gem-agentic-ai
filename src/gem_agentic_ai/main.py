"""
CLI ENTRY POINT
===============
Provides three operation modes:

  COMMAND     MODE
  demo        Single query using the LangGraph agent (default).
  chat        Multi-turn conversation with persistent memory.
  simple      Single query using the basic ReAct loop.

USAGE
  # Run with Python module syntax (recommended):
  python -m gem_agentic_ai demo
  python -m gem_agentic_ai chat
  python -m gem_agentic_ai simple --verbose

  # Or run the script directly:
  python -m gem_agentic_ai.main demo

WHY argparse?
  argparse is part of the Python standard library — zero dependencies.
  For a two-command CLI, it is perfectly sufficient. The interface is
  explicit and easy to follow for non-technical reviewers (no decorator
  magic like click/typer).

ARCHITECTURE NOTE
  Each mode function (run_demo, run_chat, run_simple_demo) constructs
  its own agent from shared components (llm, tools). This keeps the
  functions self-contained and independently testable. A shared
  helper extracts common tool-calling-output logic.
"""

import argparse
import logging
import uuid

from langchain_openai import ChatOpenAI

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.agent.simple import run_simple_agent
from gem_agentic_ai.config import settings
from gem_agentic_ai.tools import (
    calculate_multiplication,
    get_system_time,
    query_employee_database,
    web_search,
)

logger = logging.getLogger(__name__)


# ── Shared helpers ─────────────────────────────────────────────


def _configure_logging() -> None:
    """Initialise the Python logging system with the configured log level.

    Uses a consistent format that includes timestamp, log level,
    module name, and message — making it easy to trace agent behaviour.
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        # Force the root logger level in case a library overrides it.
        force=True,
    )


def _build_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance from the current application settings.

    Returns:
        A configured ChatOpenAI instance ready to be bound to tools
        or passed to build_agent().
    """
    logger.debug(
        "Creating ChatOpenAI(model=%s, temperature=%s)",
        settings.openai_model,
        settings.temperature,
    )
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.temperature,
    )


def _get_tools() -> list:
    """Return the complete tool suite available to agents.

    The order matters: put simple, reliable tools first so the LLM
    sees them before more complex ones.

    Returns:
        A list of LangChain BaseTool objects.
    """
    return [
        get_system_time,
        calculate_multiplication,
        web_search,
        query_employee_database,
    ]


def _get_tools_by_name() -> dict[str, object]:
    """Build a name-to-tool lookup for the simple ReAct loop.

    The simple agent does not use LangGraph's tool routing — it
    needs an explicit map of tool names to callable objects.

    Returns:
        Dict like {"get_system_time": <BaseTool>, "web_search": <BaseTool>, ...}
    """
    return {tool.name: tool for tool in _get_tools()}


def _extract_final_answer(events: list[dict]) -> str:
    """Extract the final AI answer text from a stream of LangGraph events.

    LangGraph's .stream() yields a sequence of events — one per graph
    node execution. We iterate through them looking for the last
    AIMessage that contains content (as opposed to tool calls).

    Args:
        events: The full list of events from agent.stream().

    Returns:
        The final answer text, or an empty string if none found.
    """
    final_answer = ""
    for event in events:
        for node_data in event.values():
            if "messages" not in node_data:
                continue
            last_msg = node_data["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                continue  # this is a tool-calling step, not the final answer
            if hasattr(last_msg, "content") and last_msg.content:
                final_answer = last_msg.content
    return final_answer


# ── Mode implementations ───────────────────────────────────────


def run_demo() -> None:
    """Single-query demo using the LangGraph agent.

    Prompts the user for one question, streams the full agent trace
    (showing every tool call and intermediate step), then prints the
    final answer.
    """
    llm = _build_llm()
    tools = _get_tools()
    agent = build_agent(llm, tools)

    # Each session gets a unique thread_id so conversations do not
    # bleed into each other (even with in-memory checkpointing).
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    question = input("👤 Enter your query: ")
    print("\n🧠 Processing (LangGraph agent)...\n")

    events: list = []
    for event in agent.stream({"messages": [("user", question)]}, config=config):
        events.append(event)
        for node_name, node_data in event.items():
            if "messages" in node_data:
                last_msg = node_data["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_name = last_msg.tool_calls[0]["name"]
                    print(f"  🔧 Tool called: {tool_name}")

    final_answer = _extract_final_answer(events)
    print(f"\n🤖 Final Answer: {final_answer or 'I could not generate a response.'}")


def run_chat() -> None:
    """Multi-turn chat mode with persistent memory across turns.

    Each user message is added to the same conversation thread, so
    the agent remembers context from earlier exchanges. Type 'exit'
    or 'quit' to end the session.
    """
    llm = _build_llm()
    tools = _get_tools()
    agent = build_agent(llm, tools)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("🤖 Multi-turn Agent Chat (type 'exit' to stop)\n")

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        events: list = []
        for event in agent.stream(
            {"messages": [("user", user_input)]}, config=config
        ):
            events.append(event)
            for node_name, node_data in event.items():
                if "messages" in node_data:
                    last_msg = node_data["messages"][-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        tool_name = last_msg.tool_calls[0]["name"]
                        print(f"  🔧 Tool called: {tool_name}")

        final_answer = _extract_final_answer(events)
        print(f"🤖 Agent: {final_answer or 'I could not generate a response.'}")
        print("-" * 50)


def run_simple_demo() -> None:
    """Single-query demo using the basic ReAct loop (no LangGraph).

    This demonstrates the same tool-use capability but without
    LangGraph's persistence, streaming, or state machine. Useful
    for comparing the two architectures side by side.
    """
    llm = _build_llm()
    tools = _get_tools()
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = _get_tools_by_name()

    question = input("👤 Enter your query: ")
    print("\n🧠 Processing (Simple ReAct loop)...\n")

    answer = run_simple_agent(llm_with_tools, question, tools_by_name)
    print(f"\n🤖 Final Answer: {answer}")


# ── CLI argument parsing ──────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments with .mode and .verbose attributes.
    """
    parser = argparse.ArgumentParser(
        description="Gem Agentic AI — LLM agent orchestration framework.",
        epilog=(
            "Examples:\n"
            "  python -m gem_agentic_ai demo\n"
            "  python -m gem_agentic_ai chat\n"
            "  python -m gem_agentic_ai simple --verbose\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="demo",
        choices=["demo", "chat", "simple"],
        help=(
            "Operation mode:\n"
            "  demo    — single query with full trace (default)\n"
            "  chat    — multi-turn conversation with memory\n"
            "  simple  — single query using the basic ReAct loop"
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for detailed traces.",
    )
    return parser.parse_args(argv)


# ── Entry point ────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Application entry point.

    Parses arguments, configures logging, and dispatches to the
    requested mode.

    Args:
        argv: Optional argument list (used by tests; None = real CLI).
    """
    args = parse_args(argv)

    # Allow --verbose to override the .env log level at runtime.
    if args.verbose:
        settings.log_level = "DEBUG"

    _configure_logging()

    logger.info("Starting Gem Agentic AI — mode=%s, verbose=%s", args.mode, args.verbose)

    # Dispatch table — maps mode names to handler functions.
    # Adding a new mode requires one entry here and one function above.
    mode_dispatch = {
        "demo": run_demo,
        "chat": run_chat,
        "simple": run_simple_demo,
    }
    mode_dispatch[args.mode]()

    logger.info("Shutting down Gem Agentic AI.")


if __name__ == "__main__":
    main()
