"""Execution-level tests for both agent implementations.

These are the behaviours A01 requires CI to exercise without credentials:

  - a real tool is dispatched and its observation is fed back to the model
  - an unknown tool is reported and not executed
  - a tool exception becomes an observation instead of crashing the run
  - turn exhaustion stops the loop with an explicit reason

No network and no model API: the model is scripted, the tools are real.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.agent.simple import run_simple_agent
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

        assert result == "The answer is 42."
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

        assert result == "I could not use that tool."
        observations = _tool_messages(model.seen[1])
        assert len(observations) == 1
        assert "unknown tool" in observations[0].content

    def test_tool_exception_is_captured_as_observation(self) -> None:
        """A raising tool must not end the run; the error goes back to the model."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("exploding_tool", {"reason": "test"})]),
                AIMessage(content="The tool failed."),
            ]
        )

        result = run_simple_agent(model, "Use the tool", {"exploding_tool": exploding_tool})

        assert result == "The tool failed."
        observations = _tool_messages(model.seen[1])
        assert len(observations) == 1
        assert "Error during tool execution" in observations[0].content
        assert "deliberate failure: test" in observations[0].content

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
        assert "allowed number of turns (3)" in result


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

    def test_graph_tool_exception_propagates(self) -> None:
        """Documents current behaviour: the LangGraph path does not recover.

        The manual loop converts an unexpected tool exception into an
        observation (tested above). LangGraph's tool node, as constructed by
        ``create_agent`` in the pinned versions, re-raises instead. Adding
        recovery there is A03 work; this test pins the difference so a change
        cannot happen silently.
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

        with pytest.raises(RuntimeError, match="deliberate failure: graph"):
            self._invoke(agent, "Use the tool")

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
