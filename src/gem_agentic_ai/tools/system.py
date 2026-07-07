"""
SYSTEM TOOLS
============
Tools that rely only on the Python standard library.
These work offline and have zero external dependencies,
making them ideal for testing the agent's tool-calling loop.

WHY separate files per domain?
  Single responsibility principle: each file owns one category of tool.
  This makes tools independently testable, importable, and swapable.
  Adding a new system tool requires no changes to web tools, etc.

ARCHITECTURE NOTE
  Every tool is decorated with @tool from langchain_core. This
  decorator converts the function into a LangChain BaseTool object
  with a standardised interface (.invoke(), .name, .args, .description).
  Both the simple ReAct loop and the LangGraph agent consume tools
  through this same interface — no tool needs to know which agent
  architecture is using it.
"""

import logging
from datetime import datetime

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_system_time(format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current date and time from the system clock.

    This is the simplest tool in the suite — it uses only Python's
    built-in datetime module. We place it first in every tool list
    so that even if external services fail, the agent can still
    answer basic questions about the current time.

    Args:
        format_string: A strftime-compatible format string.
            Defaults to ISO 8601 (YYYY-MM-DD HH:MM:SS).
            Common alternatives:
              - "%Y-%m-%d"       → "2026-07-07"
              - "%A, %B %d, %Y"  → "Tuesday, July 07, 2026"
              - "%I:%M %p"       → "04:02 PM"

    Returns:
        The formatted current date-time as a string.

    Example tool call from the LLM:
        get_system_time({"format_string": "%Y-%m-%d"})
        → "2026-07-07"
    """
    logger.debug("get_system_time called with format: %s", format_string)
    return datetime.now().strftime(format_string)


@tool
def calculate_multiplication(a: float, b: float) -> float:
    """Multiply two numbers together.

    A simple arithmetic tool. The LLM will request this when it
    needs to perform multiplication as part of a larger chain of
    reasoning (e.g., "If a product costs $45 and we need 12 units,
    what is the total cost?").

    Args:
        a: First factor (multiplicand).
        b: Second factor (multiplier).

    Returns:
        The product of a and b (a × b).

    Example tool call from the LLM:
        calculate_multiplication({"a": 45, "b": 12})
        → 540.0
    """
    logger.debug("calculate_multiplication called: %s * %s", a, b)
    return a * b
