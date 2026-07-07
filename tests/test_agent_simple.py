"""
TESTS: SIMPLE REACT AGENT
=========================
Tests the manual ReAct loop in agent/simple.py.

We mock the LLM to control its behaviour:
  - Return a direct answer (no tool calls).
  - Return tool calls, then a final answer.
  - Return multiple tool calls.
  - Exceed the maximum turn limit.

This approach tests the agent's orchestration logic without
needing an actual LLM API key.
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from gem_agentic_ai.agent.simple import run_simple_agent


class TestRunSimpleAgent:
    """Tests for the simple ReAct loop orchestration."""

    def test_direct_answer_no_tool_calls(self) -> None:
        """When the LLM returns text directly, the agent should return it immediately."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="42")

        result = run_simple_agent(mock_llm, "What is 2+2?", {})

        assert result == "42"
        # The LLM should have been invoked exactly once.
        mock_llm.invoke.assert_called_once()

    def test_single_tool_call_then_answer(self) -> None:
        """LLM calls one tool, gets a result, then produces the final answer."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            # Turn 1: LLM decides to call a tool.
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_system_time",
                        "args": {},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            # Turn 2: LLM reads the tool result and answers.
            AIMessage(content="The time is 12:00."),
        ]

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "2026-07-07 12:00:00"

        result = run_simple_agent(
            mock_llm,
            "What time is it?",
            {"get_system_time": mock_tool},
        )

        assert result == "The time is 12:00."
        mock_tool.invoke.assert_called_once_with({})
        # LLM was called twice: once for the tool decision, once for the answer.
        assert mock_llm.invoke.call_count == 2

    def test_multiple_tool_calls_in_one_turn(self) -> None:
        """LLM may request multiple tools in a single response."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            # Turn 1: LLM calls two tools.
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate_multiplication",
                        "args": {"a": 3, "b": 4},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "calculate_multiplication",
                        "args": {"a": 5, "b": 6},
                        "id": "call_2",
                        "type": "tool_call",
                    },
                ],
            ),
            # Turn 2: Final answer.
            AIMessage(content="3*4=12 and 5*6=30"),
        ]

        mock_tool = MagicMock()
        mock_tool.name = "calculate_multiplication"
        mock_tool.invoke.side_effect = [12, 30]

        result = run_simple_agent(
            mock_llm,
            "Calculate 3*4 and 5*6",
            {"calculate_multiplication": mock_tool},
        )

        assert result == "3*4=12 and 5*6=30"
        assert mock_tool.invoke.call_count == 2

    def test_unknown_tool_returns_error(self) -> None:
        """When the LLM requests a tool not in the registry, return an error message."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "nonexistent_tool",
                        "args": {},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I see the error."),
        ]

        result = run_simple_agent(mock_llm, "Do something", {})

        assert "I see the error." == result
        # The error message should mention the unknown tool.
        second_call_args = mock_llm.invoke.call_args_list[0][0][0]
        tool_msgs = [m for m in second_call_args if hasattr(m, "tool_call_id")]
        assert len(tool_msgs) > 0

    def test_tool_exception_is_caught(self) -> None:
        """If a tool raises an exception, the agent should catch it and continue."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "failing_tool",
                        "args": {},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The tool failed."),
        ]

        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = RuntimeError("Something went wrong")

        result = run_simple_agent(mock_llm, "Use the tool", {"failing_tool": mock_tool})

        assert result == "The tool failed."

    def test_max_turns_exceeded(self) -> None:
        """If the agent keeps calling tools beyond max_turns, return a timeout message."""
        mock_llm = MagicMock()
        # Always return a tool call — never a final answer.
        mock_llm.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_system_time",
                    "args": {},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "12:00:00"

        result = run_simple_agent(
            mock_llm,
            "Loop forever",
            {"get_system_time": mock_tool},
            max_turns=3,
        )

        assert "sorry" in result.lower() or "turn limit" in result.lower()
        # LLM should have been invoked max_turns times.
        assert mock_llm.invoke.call_count == 3
