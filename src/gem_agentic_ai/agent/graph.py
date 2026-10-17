"""
LANGGRAPH AGENT ORCHESTRATOR
============================
A LangGraph agent built on the framework's state machine.

WHY LangGraph over a manual loop?
  The simple ReAct loop (simple.py) works fine for single-turn demos.
  For anything more, LangGraph provides:

  1. FORMAL STATE MACHINE — Nodes (LLM, tools) and edges (conditional
     routing) are explicitly defined, making the agent's control flow
     inspectable and visualisable.

  2. PERSISTENCE — A checkpointer saves conversation state after every
     step. MemorySaver is in-process only; pass
     ``DurableMemory(...).checkpointer`` (persistence.py) to keep state in
     a local SQLite file across restarts.

  3. STREAMING — The agent yields intermediate states, allowing the
     UI to show tool calls as they happen.

  4. EXTENSIBILITY — Sub-graphs, parallel tool execution and branching
     are configuration changes rather than rewrites.

BOUNDED EXECUTION
  ``create_agent``'s tool node re-raises an unexpected tool exception and
  ends the run. Rather than patching that one path, ``build_agent`` wraps
  every tool in a ``GuardedTool``. The guard validates arguments, enforces
  timeouts and budgets, applies idempotency and approval, and returns a
  finished ``ToolMessage``. Both agent implementations therefore share one
  policy and one failure behaviour.

  The graph's own ``recursion_limit`` still bounds model steps that make no
  tool calls at all. It is a separate, complementary bound.

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
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer

from gem_agentic_ai.bounds import BudgetSnapshot
from gem_agentic_ai.guard import GuardedTool, ToolGuard
from gem_agentic_ai.models.state import AgentState

logger = logging.getLogger(__name__)


class BoundedAgent:
    """The compiled graph plus the guard that bounds its tool calls.

    The guard is reset at the start of every ``invoke`` / ``stream`` call, so
    one chat turn cannot exhaust the next turn's budget. Idempotency records
    and approval decisions live outside the budget and survive the reset.

    Any attribute not defined here is delegated to the compiled graph, so
    ``get_state``, ``update_state`` and the rest of the LangGraph API remain
    available.
    """

    def __init__(self, graph: Any, guard: ToolGuard) -> None:
        self._graph = graph
        self.guard = guard

    @property
    def name(self) -> str:
        return self._graph.name

    @property
    def budget(self) -> BudgetSnapshot:
        return self.guard.budget.snapshot()

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.guard.reset()
        return self._graph.invoke(input, config, **kwargs)

    def stream(self, input: Any, config: Any = None, **kwargs: Any):
        self.guard.reset()
        yield from self._graph.stream(input, config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.guard.reset()
        return await self._graph.ainvoke(input, config, **kwargs)

    async def astream(self, input: Any, config: Any = None, **kwargs: Any):
        self.guard.reset()
        stream = self._graph.astream(input, config, **kwargs)
        try:
            async for event in stream:
                yield event
        finally:
            # Close the graph's generator explicitly. A consumer that is
            # cancelled or closed above this wrapper otherwise leaves the
            # inner generator to the async-generator finaliser, which is not
            # prompt and, in practice, did not stop a running model node:
            # tokens kept arriving after the caller had gone. Closing it here
            # throws ``GeneratorExit`` into LangGraph, whose execution loop
            # then cancels the node tasks that were still in flight.
            await stream.aclose()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._graph, item)


def build_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str | None = None,
    checkpointer: Checkpointer | None = None,
    guard: ToolGuard | None = None,
) -> BoundedAgent:
    """Construct a LangGraph agent whose tool calls are bounded.

    Args:
        llm: A LangChain chat model instance (e.g., ChatOpenAI).
        tools: The tools the agent may call. This list is also the allowlist.
        system_prompt: Optional system message prepended to every
            conversation.
        checkpointer: Persistence backend for multi-turn memory. Defaults to
            MemorySaver, which is in-process only. Pass
            ``DurableMemory(directory).checkpointer`` for restart-safe,
            user-scoped storage.
        guard: Optional policy layer shared with another agent. A guard is
            created when omitted, and its allowlist is set from ``tools``.

    Returns:
        A ``BoundedAgent``: the compiled graph plus its guard and budget.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from gem_agentic_ai.tools import get_system_time, web_search
        >>> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        >>> agent = build_agent(llm, [get_system_time, web_search])
        >>> result = agent.invoke(
        ...     {"messages": [("user", "What time is it?")]},
        ...     {"configurable": {"thread_id": "demo-1"}},
        ... )
        >>> agent.budget.stop_reason  # None when nothing was cut short
    """
    guard = guard or ToolGuard()
    guard.restrict_to(tool.name for tool in tools)

    guarded: list[BaseTool] = [
        tool if isinstance(tool, GuardedTool) else guard.wrap(tool) for tool in tools
    ]
    checkpointer = checkpointer or MemorySaver()
    logger.info(
        "Building LangGraph agent: model=%s, tools=%d, checkpointer=%s",
        getattr(llm, "model_name", type(llm).__name__),
        len(guarded),
        type(checkpointer).__name__,
    )

    agent = create_agent(
        model=llm,
        tools=guarded,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        state_schema=AgentState,
        name="gem_agent",
    )

    logger.debug("Agent graph compiled successfully.")
    return BoundedAgent(agent, guard)
