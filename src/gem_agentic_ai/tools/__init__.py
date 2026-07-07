# Tools package — each domain (system, web, database) lives in its own module.
# New tools should be added to the appropriate module, or a new module created
# if a new domain emerges. Every tool is decorated with @tool from langchain_core,
# making them compatible with both the simple ReAct loop and the LangGraph agent.

from gem_agentic_ai.tools.system import get_system_time, calculate_multiplication
from gem_agentic_ai.tools.web import web_search
from gem_agentic_ai.tools.database import query_employee_database

__all__ = [
    "get_system_time",
    "calculate_multiplication",
    "web_search",
    "query_employee_database",
]
