# Agent package — two implementations of the ReAct agent pattern.
#
#   simple.py   — Manual thought-action-observation loop (~40 lines).
#                 Shows what LangGraph does under the hood.
#
#   graph.py    — LangGraph state machine with persistence,
#                 streaming, and formal state management.
#                 Production-grade entry point.

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.agent.simple import run_simple_agent

__all__ = [
    "build_agent",
    "run_simple_agent",
]
