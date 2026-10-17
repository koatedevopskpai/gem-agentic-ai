# Agent package — two implementations of the ReAct agent pattern.
#
#   simple.py   — Manual thought-action-observation loop.
#                 Shows what LangGraph does under the hood.
#
#   graph.py    — LangGraph state machine with persistence,
#                 streaming, and formal state management.
#
#   Both paths route tool calls through gem_agentic_ai.guard.ToolGuard,
#   so validation, timeouts, budgets, retries, idempotency and approval
#   follow one policy rather than two.

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.agent.simple import run_simple_agent

__all__ = [
    "build_agent",
    "run_simple_agent",
]
