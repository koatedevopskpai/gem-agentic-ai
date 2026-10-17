"""
SIMPLE REACT LOOP
=================
A minimal agent that manually orchestrates the thought-action-observation
loop without LangGraph or any other orchestration framework.

WHY keep this alongside the LangGraph version?
  This implementation shows exactly what a ReAct agent does under the hood:
    1. LLM "thinks" (produces text or tool calls).
    2. If tool calls → execute them → feed results back to LLM.
    3. Repeat until LLM returns plain text (the final answer).

  Readers can study this file first, then see how LangGraph formalises
  the same pattern with state machines, checkpointing, and streaming.

BOUNDED EXECUTION
  The loop itself only limits turns. Everything else — schema validation,
  the tool allowlist, per-call timeouts, call and token budgets, retries,
  idempotency and approval — comes from ``ToolGuard``, which is also used by
  the LangGraph builder. Both implementations therefore share one policy.

ARCHITECTURE NOTE
  This agent receives an LLM with tools bound and a name-to-tool mapping. It
  does NOT import or know about any concrete tool implementation.
"""

import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from gem_agentic_ai.bounds import AgentRunResult, StopReason
from gem_agentic_ai.guard import ToolGuard, ToolOutcome

logger = logging.getLogger(__name__)

# Reasons that end the run immediately rather than being fed back to the
# model. Per-tool refusals (circuit open, per-tool call limit) are not here:
# the model may still answer using what it already has.
_TERMINAL = frozenset(
    {
        StopReason.CANCELLED,
        StopReason.DEADLINE_EXCEEDED,
        StopReason.CALL_BUDGET_EXCEEDED,
        StopReason.TOKEN_BUDGET_EXCEEDED,
    }
)


def _stopped(reason: StopReason, calls: int, guard: ToolGuard) -> AgentRunResult:
    return AgentRunResult(
        answer=(
            f"Stopped: {reason.value} after {calls} tool call(s). "
            "The request could not be completed within its execution bounds."
        ),
        stop_reason=reason,
        budget=guard.budget.snapshot(),
    )


def run_simple_agent(
    llm_with_tools: Any,
    question: str,
    tools_by_name: Mapping[str, BaseTool],
    max_turns: int = 10,
    guard: ToolGuard | None = None,
) -> AgentRunResult:
    """Execute a manual ReAct (Reasoning + Act) loop under a tool policy.

    Args:
        llm_with_tools: A LangChain chat model with tools bound via
            ``.bind_tools()``.
        question: The user's input string.
        tools_by_name: The tool registry. Its keys are also the allowlist:
            a tool the model names but the registry does not contain is
            refused without being executed.
        max_turns: Maximum model turns. This bounds the model loop; the guard
            bounds tool execution separately.
        guard: Optional pre-built guard. A default guard is created from the
            registry when omitted. The guard is reset at the start of the run.

    Returns:
        An ``AgentRunResult`` carrying the final answer, the reason the loop
        stopped and a snapshot of the guard's budget and audit trail.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from gem_agentic_ai.tools import get_system_time, web_search
        >>> llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        >>> tools = [get_system_time, web_search]
        >>> run_simple_agent(
        ...     llm.bind_tools(tools),
        ...     "What time is it?",
        ...     {t.name: t for t in tools},
        ... ).answer
        'The current time is 14:30:00.'
    """
    logger.info("Starting simple ReAct agent with question: %s", question)

    guard = guard or ToolGuard()
    guard.restrict_to(tools_by_name.keys())
    guard.reset()

    messages: list = [HumanMessage(content=question)]

    for turn in range(1, max_turns + 1):
        logger.debug("ReAct turn %d/%d", turn, max_turns)

        if guard.cancelled:
            return _stopped(StopReason.CANCELLED, guard.budget.calls, guard)
        if guard.budget.remaining_seconds() <= 0:
            return _stopped(StopReason.DEADLINE_EXCEEDED, guard.budget.calls, guard)

        # Step 1: LLM THINKS — may produce a final answer or tool calls.
        response: AIMessage = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            logger.info("Agent produced final answer after %d turn(s).", turn)
            return AgentRunResult(
                answer=response.content or "",
                stop_reason=StopReason.COMPLETED,
                budget=guard.budget.snapshot(),
            )

        # Step 2: LLM ACTS — every call passes through the guard.
        tool_messages: list[ToolMessage] = []
        stop_after_turn: StopReason | None = None

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            logger.info("Calling tool: %s(%s)", tool_name, tool_args)

            outcome: ToolOutcome
            if tool_name not in tools_by_name:
                outcome = guard.reject(tool_name, detail=f"unknown tool '{tool_name}'")
            else:
                outcome = guard.execute(tools_by_name[tool_name], tool_args, call_id=tool_id)

            tool_messages.append(
                ToolMessage(
                    content=outcome.content,
                    tool_call_id=tool_id,
                    status="error" if outcome.status == "error" else "success",
                )
            )
            if outcome.stop_reason in _TERMINAL:
                stop_after_turn = outcome.stop_reason

        # Step 3: FEED BACK — give the LLM its own thoughts + tool results.
        messages.append(response)
        messages.extend(tool_messages)

        if stop_after_turn is not None:
            return _stopped(stop_after_turn, guard.budget.calls, guard)

    logger.warning("Agent reached max turns (%d) without a final answer.", max_turns)
    return AgentRunResult(
        answer=(
            "I'm sorry, I could not complete your request within the "
            f"allowed number of turns ({max_turns}). Please try a simpler query."
        ),
        stop_reason=StopReason.MAX_TURNS,
        budget=guard.budget.snapshot(),
    )
