"""End-to-end bounded-execution tests for both agent implementations.

Every test here is offline: a scripted model stands in for the provider and
the tools are real. The assertions are on what a caller can observe — the
returned stop reason, the audit trail, the model's observations and whether a
side effect happened.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.agent.simple import run_simple_agent
from gem_agentic_ai.bounds import StopReason, ToolLimits
from gem_agentic_ai.guard import ToolGuard
from gem_agentic_ai.tools.actions import (
    create_support_ticket,
    reset_ticket_ledger,
    ticket_ledger,
)
from gem_agentic_ai.tools.system import calculate_multiplication
from tests.fakes import ScriptedChatModel, tool_call

_CALLS: dict[str, int] = {}


def _count(name: str) -> int:
    _CALLS[name] = _CALLS.get(name, 0) + 1
    return _CALLS[name]


@pytest.fixture(autouse=True)
def clean_state():
    _CALLS.clear()
    reset_ticket_ledger()
    yield
    _CALLS.clear()
    reset_ticket_ledger()


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    _count("add")
    return a + b


@tool
def exploding_tool(reason: str = "boom") -> str:
    """Always raise, to exercise the failure path."""
    _count("exploding_tool")
    raise RuntimeError(f"deliberate failure: {reason}")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class AdvancingChatModel(ScriptedChatModel):
    """A scripted model that consumes wall-clock time while it 'thinks'."""

    clock: Any = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        self.clock.advance(1.0)
        return result


class CancellingChatModel(ScriptedChatModel):
    """A scripted model that asks the caller to cancel after it responds."""

    guard: Any = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        self.guard.cancel()
        return result


def _tool_messages(messages: list) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def _looping_calls(name: str, args: dict, count: int) -> list[AIMessage]:
    return [
        AIMessage(content="", tool_calls=[tool_call(name, args, f"call_{i}")])
        for i in range(count)
    ]


class TestSimpleLoopBounds:
    def test_call_budget_stops_the_loop(self) -> None:
        guard = ToolGuard(ToolLimits(max_total_calls=2, max_calls_per_tool=2))
        model = ScriptedChatModel(
            responses=_looping_calls("add", {"a": 1, "b": 1}, 3)
        )

        result = run_simple_agent(
            model, "Loop forever", {"add": add}, max_turns=5, guard=guard
        )

        assert result.stop_reason is StopReason.CALL_BUDGET_EXCEEDED
        assert result.budget.stop_reason is StopReason.CALL_BUDGET_EXCEEDED
        assert result.budget.calls == 2
        assert _CALLS["add"] == 2
        assert len(model.seen) == 3
        assert "call_budget_exceeded" in result.answer

    def test_circuit_breaker_stops_repeated_failures(self) -> None:
        guard = ToolGuard(
            ToolLimits(
                max_total_calls=10,
                max_calls_per_tool=10,
                max_attempts_per_call=1,
                max_consecutive_failures=2,
            )
        )
        model = ScriptedChatModel(
            responses=_looping_calls("exploding_tool", {"reason": "loop"}, 4)
        )

        result = run_simple_agent(
            model, "Keep trying", {"exploding_tool": exploding_tool}, max_turns=4, guard=guard
        )

        # The loop ends on its own turn limit, but the budget records why the
        # tool was cut off. A completed answer does not erase that.
        assert result.stop_reason is StopReason.MAX_TURNS
        assert result.budget.stop_reason is StopReason.CIRCUIT_OPEN
        assert _CALLS["exploding_tool"] == 2

    def test_cancellation_stops_before_the_tool_runs(self) -> None:
        guard = ToolGuard()
        model = CancellingChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("add", {"a": 1, "b": 1})]),
            ],
            guard=guard,
        )

        result = run_simple_agent(model, "Use the tool", {"add": add}, guard=guard)

        assert result.stop_reason is StopReason.CANCELLED
        assert _CALLS.get("add", 0) == 0
        assert guard.cancelled is True

    def test_deadline_stops_before_the_tool_runs(self) -> None:
        clock = FakeClock()
        guard = ToolGuard(ToolLimits(deadline_seconds=0.5), clock=clock)
        model = AdvancingChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("add", {"a": 1, "b": 1})]),
            ],
            clock=clock,
        )

        result = run_simple_agent(model, "Use the tool", {"add": add}, guard=guard)

        assert result.stop_reason is StopReason.DEADLINE_EXCEEDED
        assert result.budget.stop_reason is StopReason.DEADLINE_EXCEEDED
        assert _CALLS.get("add", 0) == 0

    def test_invalid_arguments_do_not_execute(self) -> None:
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[tool_call("add", {"a": "not a number", "b": 2})],
                ),
                AIMessage(content="That call was invalid."),
            ]
        )

        result = run_simple_agent(model, "Add", {"add": add})

        observations = _tool_messages(model.seen[1])
        assert result.answer == "That call was invalid."
        assert observations[0].status == "error"
        assert "invalid arguments" in observations[0].content
        assert _CALLS.get("add", 0) == 0

    def test_completed_run_records_no_stop_reason(self) -> None:
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("add", {"a": 2, "b": 3})]),
                AIMessage(content="5"),
            ]
        )

        result = run_simple_agent(model, "Add 2 and 3", {"add": add})

        assert result.stop_reason is StopReason.COMPLETED
        assert result.budget.stop_reason is None
        assert result.budget.calls == 1


class TestGraphBounds:
    def _invoke(self, agent, question: str, thread_id: str = "bounded-thread"):
        return agent.invoke(
            {"messages": [("user", question)]},
            {"configurable": {"thread_id": thread_id}},
        )

    def test_call_budget_refuses_and_records(self) -> None:
        guard = ToolGuard(ToolLimits(max_total_calls=2, max_calls_per_tool=2))
        model = ScriptedChatModel(
            responses=[
                *_looping_calls("calculate_multiplication", {"a": 1, "b": 1}, 3),
                AIMessage(content="I used what I had."),
            ]
        )
        agent = build_agent(model, [calculate_multiplication], guard=guard)

        result = self._invoke(agent, "Loop forever")
        observations = _tool_messages(result["messages"])

        assert len(observations) == 3
        assert observations[0].status == "success"
        assert observations[1].status == "success"
        assert observations[2].status == "error"
        assert "call_budget_exceeded" in observations[2].content
        assert guard.budget.calls == 2
        assert guard.budget.stop_reason is StopReason.CALL_BUDGET_EXCEEDED
        assert result["messages"][-1].content == "I used what I had."

    def test_invalid_arguments_do_not_execute(self) -> None:
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[tool_call("add", {"a": "not a number", "b": 2})],
                ),
                AIMessage(content="That call was invalid."),
            ]
        )
        agent = build_agent(model, [add])

        result = self._invoke(agent, "Add")
        observations = _tool_messages(result["messages"])

        assert observations[0].status == "error"
        assert "invalid arguments" in observations[0].content
        assert _CALLS.get("add", 0) == 0
        assert result["messages"][-1].content == "That call was invalid."

    def test_consequential_action_requires_approval(self) -> None:
        guard = ToolGuard(
            ToolLimits(consequential_tools=frozenset({"create_support_ticket"}))
        )
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "create_support_ticket",
                            {"summary": "printer offline", "idempotency_key": "ticket-1"},
                        )
                    ],
                ),
                AIMessage(content="Awaiting approval."),
            ]
        )
        agent = build_agent(model, [create_support_ticket], guard=guard)

        result = self._invoke(agent, "Open a ticket")
        observations = _tool_messages(result["messages"])

        assert observations[0].status == "error"
        assert "approval required" in observations[0].content
        assert ticket_ledger() == ()
        assert guard.approvals.pending == {
            ("create_support_ticket", "ticket-1")
        }

    def test_replay_produces_one_logical_result(self) -> None:
        guard = ToolGuard(
            ToolLimits(consequential_tools=frozenset({"create_support_ticket"}))
        )
        guard.approvals.approve("create_support_ticket", "ticket-1")
        call = tool_call(
            "create_support_ticket",
            {"summary": "printer offline", "idempotency_key": "ticket-1"},
        )
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[call]),
                AIMessage(content="Created."),
                AIMessage(content="", tool_calls=[call]),
                AIMessage(content="Already created."),
            ]
        )
        agent = build_agent(model, [create_support_ticket], guard=guard)

        self._invoke(agent, "Open a ticket", thread_id="run-1")
        self._invoke(agent, "Open a ticket", thread_id="run-2")

        assert len(ticket_ledger()) == 1
        assert len(model.seen) == 4

    def test_budget_is_reset_between_turns(self) -> None:
        guard = ToolGuard(ToolLimits(max_total_calls=1, max_calls_per_tool=1))
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[tool_call("add", {"a": 1, "b": 1})]),
                AIMessage(content="First answer."),
                AIMessage(content="", tool_calls=[tool_call("add", {"a": 2, "b": 2})]),
                AIMessage(content="Second answer."),
            ]
        )
        agent = build_agent(model, [add], guard=guard)

        first = self._invoke(agent, "First", thread_id="turn-1")
        second = self._invoke(agent, "Second", thread_id="turn-2")

        assert _tool_messages(first["messages"])[0].status == "success"
        assert _tool_messages(second["messages"])[0].status == "success"
        # One call per run: each invoke starts a fresh budget.
        assert guard.budget.calls == 1

    def test_graph_returns_an_error_observation_for_an_unexpected_exception(self) -> None:
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[tool_call("exploding_tool", {"reason": "graph"})],
                ),
                AIMessage(content="The tool failed."),
            ]
        )
        agent = build_agent(model, [exploding_tool])

        result = self._invoke(agent, "Use the tool")
        observations = _tool_messages(result["messages"])

        assert observations[0].status == "error"
        assert "Error during tool execution" in observations[0].content
        assert result["messages"][-1].content == "The tool failed."
        assert agent.budget.calls == 1
