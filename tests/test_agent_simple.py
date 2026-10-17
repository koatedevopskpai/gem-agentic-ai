"""
TESTS: SIMPLE REACT AGENT
=========================
Tests the manual ReAct loop in agent/simple.py.

The model is scripted so its behaviour is exact:
  - Return a direct answer (no tool calls).
  - Return tool calls, then a final answer.
  - Return multiple tool calls.
  - Exceed the maximum turn limit.

The tools are real ``@tool`` objects, so the guard's schema validation is
exercised rather than bypassed by a mock.
"""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from gem_agentic_ai.agent.simple import run_simple_agent
from gem_agentic_ai.bounds import StopReason
from gem_agentic_ai.tools.system import calculate_multiplication
from tests.fakes import ScriptedChatModel, tool_call


@tool
def fixed_clock() -> str:
    """Return a fixed timestamp."""
    return "2026-07-07 12:00:00"


@tool
def failing_tool() -> str:
    """Always raise, to exercise the exception path."""
    raise RuntimeError("Something went wrong")


def _tool_messages(messages: list) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


class TestRunSimpleAgent:
    """Tests for the simple ReAct loop orchestration."""

    def test_direct_answer_no_tool_calls(self) -> None:
        """When the LLM returns text directly, the agent should return it immediately."""
        model = ScriptedChatModel(responses=[AIMessage(content="42")])

        result = run_simple_agent(model, "What is 2+2?", {})

        assert result.answer == "42"
        assert result.stop_reason is StopReason.COMPLETED
        # The LLM should have been invoked exactly once.
        assert len(model.seen) == 1

    def test_single_tool_call_then_answer(self) -> None:
        """LLM calls one tool, gets a result, then produces the final answer."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("fixed_clock", {})]),
                AIMessage(content="The time is 12:00."),
            ]
        )

        result = run_simple_agent(model, "What time is it?", {"fixed_clock": fixed_clock})

        assert result.answer == "The time is 12:00."
        observations = _tool_messages(model.seen[1])
        assert len(observations) == 1
        assert observations[0].content == "2026-07-07 12:00:00"
        assert observations[0].status == "success"
        # LLM was called twice: once for the tool decision, once for the answer.
        assert len(model.seen) == 2

    def test_multiple_tool_calls_in_one_turn(self) -> None:
        """LLM may request multiple tools in a single response."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("calculate_multiplication", {"a": 3, "b": 4}, "call_1"),
                        tool_call("calculate_multiplication", {"a": 5, "b": 6}, "call_2"),
                    ],
                ),
                AIMessage(content="3*4=12 and 5*6=30"),
            ]
        )

        result = run_simple_agent(
            model,
            "Calculate 3*4 and 5*6",
            {"calculate_multiplication": calculate_multiplication},
        )

        assert result.answer == "3*4=12 and 5*6=30"
        observations = _tool_messages(model.seen[1])
        assert [float(m.content) for m in observations] == [12.0, 30.0]
        assert result.budget.calls == 2

    def test_unknown_tool_returns_error(self) -> None:
        """When the LLM requests a tool not in the registry, return an error message."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("nonexistent_tool")]),
                AIMessage(content="I see the error."),
            ]
        )

        result = run_simple_agent(model, "Do something", {})

        assert result.answer == "I see the error."
        observations = _tool_messages(model.seen[1])
        assert len(observations) == 1
        assert observations[0].status == "error"
        assert "unknown tool" in observations[0].content

    def test_tool_exception_is_caught(self) -> None:
        """If a tool raises an exception, the agent should catch it and continue."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("failing_tool")]),
                AIMessage(content="The tool failed."),
            ]
        )

        result = run_simple_agent(model, "Use the tool", {"failing_tool": failing_tool})

        assert result.answer == "The tool failed."
        observations = _tool_messages(model.seen[1])
        assert observations[0].status == "error"
        assert "Something went wrong" in observations[0].content

    def test_max_turns_exceeded(self) -> None:
        """If the agent keeps calling tools beyond max_turns, return a timeout message."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("fixed_clock", {})])
                for _ in range(3)
            ]
        )

        result = run_simple_agent(
            model,
            "Loop forever",
            {"fixed_clock": fixed_clock},
            max_turns=3,
        )

        assert result.stop_reason is StopReason.MAX_TURNS
        assert "allowed number of turns (3)" in result.answer
        # LLM should have been invoked max_turns times.
        assert len(model.seen) == 3
