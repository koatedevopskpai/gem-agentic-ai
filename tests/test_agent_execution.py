"""Execution-level tests for both agent implementations.

These are the behaviours A01 requires CI to exercise without credentials:

  - a real tool is dispatched and its observation is fed back to the model
  - an unknown tool is reported and not executed
  - a tool exception becomes an observation instead of crashing the run,
    on both the manual and the LangGraph path
  - turn exhaustion stops the loop with an explicit reason

No network and no model API: the model is scripted, the tools are real.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.agent.simple import run_simple_agent
from gem_agentic_ai.bounds import StopReason
from gem_agentic_ai.tools.system import calculate_multiplication
from tests.fakes import ScriptedChatModel, tool_call


@tool
def exploding_tool(reason: str = "boom") -> str:
    """Always raise, to exercise the tool-exception path."""
    raise RuntimeError(f"deliberate failure: {reason}")


def _tool_messages(messages: list) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


class TestSimpleLoopExecution:
    """The manual ReAct loop dispatches real tools and reads observations."""

    def test_real_tool_dispatch_feeds_observation_back(self) -> None:
        """6 * 7 must be computed by the real tool and returned to the model."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[tool_call("calculate_multiplication", {"a": 6, "b": 7})]
                ),
                AIMessage(content="The answer is 42."),
            ]
        )

        result = run_simple_agent(
            model,
            "What is 6 times 7?",
            {"calculate_multiplication": calculate_multiplication},
        )

        assert result.answer == "The answer is 42."
        assert result.stop_reason is StopReason.COMPLETED
        # The second model call must include the tool observation.
        observations = _tool_messages(model.seen[1])
        assert len(observations) == 1
        assert float(observations[0].content) == 42.0
        assert observations[0].tool_call_id == "call_1"

    def test_unknown_tool_is_reported_not_executed(self) -> None:
        """A tool name outside the registry returns an error observation."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("does_not_exist", {"x": 1})]),
                AIMessage(content="I could not use that tool."),
            ]
        )

        result = run_simple_agent(model, "Do something", {})

        assert result.answer == "I could not use that tool."
        observations = _tool_messages(model.seen[1])
        assert len(observations) == 1
        assert "unknown tool" in observations[0].content
        assert observations[0].status == "error"

    def test_tool_exception_is_captured_as_observation(self) -> None:
        """A raising tool must not end the run; the error goes back to the model."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("exploding_tool", {"reason": "test"})]),
                AIMessage(content="The tool failed."),
            ]
        )

        result = run_simple_agent(model, "Use the tool", {"exploding_tool": exploding_tool})

        assert result.answer == "The tool failed."
        observations = _tool_messages(model.seen[1])
        assert len(observations) == 1
        assert "Error during tool execution" in observations[0].content
        assert "deliberate failure: test" in observations[0].content
        assert observations[0].status == "error"
        assert result.budget.calls == 1

    def test_turn_exhaustion_stops_and_reports(self) -> None:
        """A model that never stops calling tools is cut off after max_turns."""
        looping = AIMessage(
            content="", tool_calls=[tool_call("calculate_multiplication", {"a": 1, "b": 1})]
        )
        model = ScriptedChatModel(responses=[looping] * 3)

        result = run_simple_agent(
            model,
            "Loop forever",
            {"calculate_multiplication": calculate_multiplication},
            max_turns=3,
        )

        assert len(model.seen) == 3
        assert result.stop_reason is StopReason.MAX_TURNS
        assert "allowed number of turns (3)" in result.answer


class TestLangGraphExecution:
    """The LangGraph agent dispatches tools through its own tool node."""

    def _invoke(self, agent, question: str = "What is 6 times 7?"):
        return agent.invoke(
            {"messages": [("user", question)]},
            {"configurable": {"thread_id": "test-thread"}},
        )

    def test_graph_dispatches_real_tool_and_returns_observation(self) -> None:
        """The tool node runs the real tool and passes the result to the model."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[tool_call("calculate_multiplication", {"a": 6, "b": 7})]
                ),
                AIMessage(content="The answer is 42."),
            ]
        )
        agent = build_agent(model, [calculate_multiplication])

        result = self._invoke(agent)
        messages = result["messages"]

        observations = _tool_messages(messages)
        assert len(observations) == 1
        assert float(observations[0].content) == 42.0
        assert messages[-1].content == "The answer is 42."
        # The observation also reached the model on its second turn.
        assert float(_tool_messages(model.seen[-1])[0].content) == 42.0

    def test_graph_unknown_tool_is_reported_not_executed(self) -> None:
        """The graph's tool node returns an error observation for an unknown tool."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("does_not_exist", {"x": 1})]),
                AIMessage(content="I could not use that tool."),
            ]
        )
        agent = build_agent(model, [calculate_multiplication])

        result = self._invoke(agent, "Do something")
        observations = _tool_messages(result["messages"])

        assert len(observations) == 1
        assert "is not a valid tool" in observations[0].content
        assert observations[0].status == "error"
        assert result["messages"][-1].content == "I could not use that tool."

    def test_graph_tool_exception_becomes_observation(self) -> None:
        """An unexpected tool exception must not end the graph run.

        A01 pinned the opposite behaviour: the manual loop recovered while
        LangGraph re-raised. A03 routes both paths through ``ToolGuard``, so
        the exception is now an error observation on both. This test replaces
        the divergence test and pins the shared behaviour.
        """
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[tool_call("exploding_tool", {"reason": "graph"})]
                ),
                AIMessage(content="The tool failed."),
            ]
        )
        agent = build_agent(model, [exploding_tool])

        result = self._invoke(agent, "Use the tool")
        observations = _tool_messages(result["messages"])

        assert len(observations) == 1
        assert observations[0].status == "error"
        assert "Error during tool execution" in observations[0].content
        assert "deliberate failure: graph" in observations[0].content
        assert result["messages"][-1].content == "The tool failed."
        assert len(model.seen) == 2

    def test_graph_recursion_limit_stops_a_looping_model(self) -> None:
        """A model that never answers cannot run indefinitely."""
        looping = [
            AIMessage(
                content="",
                tool_calls=[tool_call("calculate_multiplication", {"a": 1, "b": 1}, f"c{i}")],
            )
            for i in range(20)
        ]
        model = ScriptedChatModel(responses=looping)
        agent = build_agent(model, [calculate_multiplication])

        with pytest.raises(GraphRecursionError):
            agent.invoke(
                {"messages": [("user", "Loop forever")]},
                {
                    "configurable": {"thread_id": "recursion-thread"},
                    "recursion_limit": 5,
                },
            )
