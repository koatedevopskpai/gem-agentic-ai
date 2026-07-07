"""
SIMPLE REACT LOOP
=================
A minimal agent that manually orchestrates the thought-action-observation
loop without LangGraph or any other orchestration framework.

WHY keep this alongside the LangGraph version?
  This ~50-line implementation shows exactly what a ReAct agent does
  under the hood:
    1. LLM "thinks" (produces text or tool calls).
    2. If tool calls → execute them → feed results back to LLM.
    3. Repeat until LLM returns plain text (the final answer).

  Readers can study this file first, then see how LangGraph formalises
  the same pattern with state machines, checkpointing, and streaming.
  It is deliberately kept as a pedagogical artifact.

ARCHITECTURE NOTE
  This agent receives an llm_with_tools (an LLM with .bind_tools()
  applied) and a tools_by_name dict. It does NOT import or know
  about any concrete tool implementation — complete decoupling.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableBinding

logger = logging.getLogger(__name__)


def run_simple_agent(
    llm_with_tools: RunnableBinding,
    question: str,
    tools_by_name: dict[str, object],
    max_turns: int = 10,
) -> str:
    """Execute a manual ReAct (Reasoning + Act) loop.

    Args:
        llm_with_tools: A LangChain LLM with tools bound via .bind_tools().
        question: The user's input string (e.g., "What is the date and weather?").
        tools_by_name: A dict mapping tool names to callable tool objects
                       (e.g., {"get_system_time": get_system_time}).
        max_turns: Safety limit to prevent infinite loops. Default 10.

    Returns:
        The LLM's final text answer, or an error message if the turn
        limit is exceeded.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from gem_agentic_ai.tools import get_system_time, web_search
        >>> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        >>> tools = [get_system_time, web_search]
        >>> llm_with = llm.bind_tools(tools)
        >>> tools_map = {t.name: t for t in tools}
        >>> run_simple_agent(llm_with, "What time is it?", tools_map)
        'The current time is 14:30:00.'
    """
    logger.info("Starting simple ReAct agent with question: %s", question)

    # ── Build the conversation context ────────────────────────
    messages: list = [HumanMessage(content=question)]

    for turn in range(1, max_turns + 1):
        logger.debug("ReAct turn %d/%d", turn, max_turns)

        # Step 1: LLM THINKS — may produce a final answer or tool calls.
        response: AIMessage = llm_with_tools.invoke(messages)

        # No tool calls means the LLM is ready with a final answer.
        if not response.tool_calls:
            logger.info("Agent produced final answer after %d turn(s).", turn)
            return response.content or ""

        # Step 2: LLM ACTS — execute each requested tool.
        tool_messages: list[ToolMessage] = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            logger.info("Calling tool: %s(%s)", tool_name, tool_args)

            if tool_name not in tools_by_name:
                logger.warning("Unknown tool requested: %s", tool_name)
                result = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    result = tools_by_name[tool_name].invoke(tool_args)
                except Exception as e:
                    logger.exception("Tool %s raised an exception: %s", tool_name, e)
                    result = f"Error during tool execution: {e}"

            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )

        # Step 3: FEED BACK — give the LLM its own thoughts + tool results.
        messages.append(response)
        messages.extend(tool_messages)
        # Loop back to Step 1 for another round of thinking.

    # ── Safety exit ───────────────────────────────────────────
    logger.warning("Agent reached max turns (%d) without a final answer.", max_turns)
    return (
        "I'm sorry, I could not complete your request within the "
        f"allowed number of turns ({max_turns}). Please try a simpler query."
    )
