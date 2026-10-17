# Tools package — each domain (system, web, database) lives in its own module.
# New tools should be added to the appropriate module, or a new module created
# if a new domain emerges. Every tool is decorated with @tool from langchain_core,
# making them compatible with both the simple ReAct loop and the LangGraph agent.

from gem_agentic_ai.tools.actions import create_support_ticket
from gem_agentic_ai.tools.database import query_employee_database
from gem_agentic_ai.tools.system import calculate_multiplication, get_system_time
from gem_agentic_ai.tools.web import web_search

__all__ = [
    "get_system_time",
    "calculate_multiplication",
    "web_search",
    "query_employee_database",
    # Consequential demo action. It is intentionally not part of the default
    # CLI tool suite: it must be registered with a guard whose limits list it
    # in `consequential_tools`, so it cannot run without approval.
    "create_support_ticket",
]
