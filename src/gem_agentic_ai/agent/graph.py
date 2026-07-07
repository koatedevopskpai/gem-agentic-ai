"""
LANGGRAPH AGENT ORCHESTRATOR
============================
A production-grade agent built on LangGraph's state machine.

WHY LangGraph over a manual loop?
  The simple ReAct loop (simple.py) works fine for single-turn demos.
  For anything more, LangGraph provides:

  1. FORMAL STATE MACHINE — Nodes (LLM, tools) and edges (conditional
     routing) are explicitly defined, making the agent's control flow
     inspectable and visualisable.

  2. PERSISTENCE — MemorySaver checkpoints conversation state after
     every step. Multi-turn chat survives server restarts (if a
     database-backed checkpointer is used).

  3. STREAMING — The agent yields intermediate states, allowing the
     UI to show tool calls as they happen (typewriter effect, tool
     status updates).

  4. EXTENSIBILITY — Human-in-the-loop, sub-graphs, parallel tool
     execution, and branching conversations are all configuration
     changes rather than rewrites.

STATE MACHINE FLOW
  ┌──────────┐     ┌───────────┐     ┌───────────┐
  │  USER    │ ──▶ │   LLM     │ ──▶ │   TOOL    │
  │ INPUT    │     │ (THINK)   │     │ (ACTUATE) │ ──┐
  └──────────┘     └───────────┘     └───────────┘   │
        ▲                                             │
        └─────────────────────────────────────────────┘
               (loop until LLM returns final answer)

USAGE
  from gem_agentic_ai.agent.graph import build_agent

  agent = build_agent(llm, tools)
  for event in agent.stream({"messages": [("user", "Hello?")]}, config):
      # process each streaming event
      pass
"""

import logging

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer

from gem_agentic_ai.models.state import AgentState

logger = logging.getLogger(__name__)


def build_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str | None = None,
    checkpointer: Checkpointer | None = None,
) -> object:
    """Construct and compile a LangGraph agent.

    Under the hood, create_agent builds a StateGraph with two nodes:
      1. An LLM-calling node that decides whether to answer or use a tool.
      2. A tool-execution node that runs the requested tool.

    A conditional edge routes from the tool node back to the LLM node
    (if more tools are needed) or to the END node (if the LLM produced
    a final answer).

    Args:
        llm: A LangChain chat model instance (e.g., ChatOpenAI).
        tools: The list of tools the agent may call during reasoning.
        system_prompt: Optional system message prepended to every
                       conversation. Use this to set the agent's persona
                       or behavioural constraints.
        checkpointer: Persistence backend for multi-turn memory.
                      Defaults to MemorySaver (in-memory).
                      Swap for PostgresSaver, SqliteSaver, etc.

    Returns:
        A CompiledStateGraph ready for .invoke() or .stream() calls.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from gem_agentic_ai.tools import get_system_time, web_search
        >>> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        >>> agent = build_agent(llm, [get_system_time, web_search])
        >>> result = agent.invoke(
        ...     {"messages": [("user", "What time is it?")]},
        ...     {"configurable": {"thread_id": "demo-1"}},
        ... )
    """
    checkpointer = checkpointer or MemorySaver()
    logger.info(
        "Building LangGraph agent: model=%s, tools=%d, checkpointer=%s",
        getattr(llm, "model_name", type(llm).__name__),
        len(tools),
        type(checkpointer).__name__,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        state_schema=AgentState,
        name="gem_agent",
    )

    logger.debug("Agent graph compiled successfully.")
    return agent
