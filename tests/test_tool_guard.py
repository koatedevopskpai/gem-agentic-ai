"""Tests for the tool guard: validation, timeouts, retries, idempotency.

Every test asserts on an observable effect (a call counter, a ledger, a
recorded status) rather than on an internal attribute, so the behaviour is
pinned rather than the implementation.
"""

import time
from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool

from gem_agentic_ai.bounds import CallStatus, StopReason, ToolLimits
from gem_agentic_ai.guard import (
    ApprovalStore,
    RetryableToolError,
    ToolGuard,
    action_key,
)
from gem_agentic_ai.idempotency import SqliteIdempotencyStore
from gem_agentic_ai.tools.actions import (
    create_support_ticket,
    reset_ticket_ledger,
    ticket_ledger,
)

# Per-test call counters. The reset fixture keeps tests independent.
_CALLS: dict[str, int] = {}


def _reset_state() -> None:
    _CALLS.clear()
    reset_ticket_ledger()


@pytest.fixture(autouse=True)
def clean_state():
    _reset_state()
    yield
    _reset_state()


def _count(name: str) -> int:
    _CALLS[name] = _CALLS.get(name, 0) + 1
    return _CALLS[name]


@tool
def recorder(value: int) -> str:
    """Record that it was called and echo the value."""
    _count("recorder")
    return f"recorded {value}"


@tool
def exploding() -> str:
    """Always raise an unexpected exception."""
    _count("exploding")
    raise RuntimeError("boom")


@tool
def transient(failures: int = 1) -> str:
    """Raise a retryable error a fixed number of times, then succeed."""
    attempt = _count("transient")
    if attempt <= failures:
        raise RetryableToolError("temporary outage")
    return "recovered"


@tool
def slow(seconds: float = 1.0) -> str:
    """Sleep before returning, to exercise the per-call timeout."""
    _count("slow")
    time.sleep(seconds)
    return "slow done"


class LegacyTool(BaseTool):
    """A BaseTool that never declared an argument schema."""

    name: str = "legacy"
    description: str = "no schema"

    def _run(self, value: int = 0) -> str:
        _count("legacy")
        return "legacy ran"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestActionKey:
    def test_explicit_key_wins(self) -> None:
        key = action_key(
            "create_support_ticket",
            {"idempotency_key": "order-42", "summary": "help"},
            "call_1",
        )
        assert key == "order-42"

    def test_call_id_is_used_when_present(self) -> None:
        assert action_key("recorder", {"value": 1}, "call_7") == "call:call_7"

    def test_arguments_hash_falls_back(self) -> None:
        first = action_key("recorder", {"value": 1})
        second = action_key("recorder", {"value": 1})
        third = action_key("recorder", {"value": 2})

        assert first == second
        assert first != third


class TestValidation:
    def test_valid_arguments_execute(self) -> None:
        guard = ToolGuard()

        outcome = guard.execute(recorder, {"value": 3})

        assert outcome.status == "ok"
        assert outcome.content == "recorded 3"
        assert _CALLS["recorder"] == 1

    def test_invalid_arguments_do_not_execute(self) -> None:
        guard = ToolGuard()

        outcome = guard.execute(recorder, {"value": "not an integer"})

        assert outcome.status == "error"
        assert outcome.error_type == "invalid_arguments"
        assert _CALLS.get("recorder", 0) == 0
        assert guard.budget.snapshot().records[-1].status is CallStatus.REFUSED

    def test_missing_required_argument_is_refused(self) -> None:
        guard = ToolGuard()

        outcome = guard.execute(recorder, {})

        assert outcome.error_type == "invalid_arguments"
        assert _CALLS.get("recorder", 0) == 0

    def test_non_mapping_arguments_are_refused(self) -> None:
        guard = ToolGuard()
        bad_args: Any = ["value"]

        outcome = guard.execute(recorder, bad_args)

        assert outcome.error_type == "invalid_arguments"

    def test_tool_without_schema_is_refused(self) -> None:
        guard = ToolGuard()

        outcome = guard.execute(LegacyTool(), {"value": 1})

        assert outcome.error_type == "unvalidated_tool"
        assert _CALLS.get("legacy", 0) == 0

    def test_oversized_arguments_are_refused(self) -> None:
        guard = ToolGuard(ToolLimits(max_argument_bytes=8))

        outcome = guard.execute(recorder, {"value": 123456})

        assert outcome.error_type == "arguments_too_large"
        assert _CALLS.get("recorder", 0) == 0

    def test_allowlist_refuses_other_tools(self) -> None:
        guard = ToolGuard()
        guard.restrict_to({"recorder"})

        outcome = guard.execute(exploding, {})

        assert outcome.error_type == "not_allowed"
        assert _CALLS.get("exploding", 0) == 0

    def test_reject_records_an_unknown_tool(self) -> None:
        guard = ToolGuard()

        outcome = guard.reject("ghost", detail="unknown tool 'ghost'")

        assert outcome.status == "error"
        assert outcome.error_type == "unknown_tool"
        assert guard.budget.snapshot().calls == 1


class TestTimeouts:
    def test_slow_tool_is_cut_off(self) -> None:
        guard = ToolGuard(
            ToolLimits(per_call_timeout_seconds=0.05, max_attempts_per_call=1)
        )

        started = time.monotonic()
        outcome = guard.execute(slow, {"seconds": 0.4})
        elapsed = time.monotonic() - started

        assert outcome.status == "error"
        assert outcome.error_type == "ToolTimeoutError"
        assert elapsed < 0.35
        assert guard.budget.snapshot().records[-1].status is CallStatus.TIMEOUT

    def test_consequential_timeout_is_not_retried(self) -> None:
        args = {"seconds": 0.4}
        approvals = ApprovalStore()
        approvals.approve("slow", action_key("slow", args))
        guard = ToolGuard(
            ToolLimits(
                per_call_timeout_seconds=0.05,
                max_attempts_per_call=3,
                consequential_tools=frozenset({"slow"}),
            ),
            approvals=approvals,
        )

        outcome = guard.execute(slow, args)

        assert outcome.error_type == "ToolTimeoutError"
        assert _CALLS["slow"] == 1
        assert guard.budget.snapshot().records[-1].attempts == 1


class TestRetries:
    def test_retryable_error_is_retried(self) -> None:
        guard = ToolGuard(ToolLimits(max_attempts_per_call=2))

        outcome = guard.execute(transient, {"failures": 1})

        assert outcome.status == "ok"
        assert outcome.content == "recovered"
        assert _CALLS["transient"] == 2
        assert guard.budget.snapshot().records[-1].attempts == 2

    def test_retryable_error_stops_at_the_limit(self) -> None:
        guard = ToolGuard(ToolLimits(max_attempts_per_call=2))

        outcome = guard.execute(transient, {"failures": 5})

        assert outcome.status == "error"
        assert outcome.error_type == "RetryableToolError"
        assert _CALLS["transient"] == 2

    def test_unknown_exception_is_not_retried(self) -> None:
        guard = ToolGuard(ToolLimits(max_attempts_per_call=3))

        outcome = guard.execute(exploding, {})

        assert outcome.status == "error"
        assert outcome.error_type == "RuntimeError"
        assert _CALLS["exploding"] == 1


class TestCircuitBreaker:
    def test_open_circuit_refuses_further_calls(self) -> None:
        guard = ToolGuard(
            ToolLimits(max_consecutive_failures=2, max_attempts_per_call=1)
        )

        assert guard.execute(exploding, {}).status == "error"
        assert guard.execute(exploding, {}).status == "error"
        refused = guard.execute(exploding, {})

        assert _CALLS["exploding"] == 2
        assert refused.error_type == "budget"
        assert refused.stop_reason is StopReason.CIRCUIT_OPEN
        assert guard.budget.stop_reason is StopReason.CIRCUIT_OPEN

class TestIdempotency:
    def test_same_action_key_executes_once(self) -> None:
        guard = ToolGuard()

        first = guard.execute(
            create_support_ticket,
            {"summary": "printer offline", "idempotency_key": "ticket-1"},
            call_id="call_1",
        )
        second = guard.execute(
            create_support_ticket,
            {"summary": "printer offline", "idempotency_key": "ticket-1"},
            call_id="call_2",
        )

        assert first.status == second.status == "ok"
        assert second.cached is True
        assert len(ticket_ledger()) == 1

    def test_repeated_call_id_is_cached(self) -> None:
        guard = ToolGuard()

        guard.execute(recorder, {"value": 1}, call_id="call_9")
        cached = guard.execute(recorder, {"value": 1}, call_id="call_9")

        assert cached.cached is True
        assert _CALLS["recorder"] == 1

    def test_completed_action_survives_a_new_guard(self, tmp_path) -> None:
        path = tmp_path / "idempotency.sqlite"

        first = ToolGuard(idempotency=SqliteIdempotencyStore(path))
        first.execute(
            create_support_ticket,
            {"summary": "printer offline", "idempotency_key": "ticket-1"},
        )

        # A fresh process would build a fresh guard and store.
        second = ToolGuard(idempotency=SqliteIdempotencyStore(path))
        outcome = second.execute(
            create_support_ticket,
            {"summary": "printer offline", "idempotency_key": "ticket-1"},
        )

        assert outcome.cached is True
        assert len(ticket_ledger()) == 1


class TestApproval:
    def test_consequential_tool_requires_approval(self) -> None:
        guard = ToolGuard(ToolLimits(consequential_tools=frozenset({"create_support_ticket"})))

        refused = guard.execute(
            create_support_ticket,
            {"summary": "printer offline", "idempotency_key": "ticket-1"},
        )

        assert refused.error_type == "approval_required"
        assert refused.status == "error"
        assert ticket_ledger() == ()
        assert guard.approvals.pending == {("create_support_ticket", "ticket-1")}
        assert guard.budget.snapshot().records[-1].status is CallStatus.APPROVAL_REQUIRED

    def test_approved_action_executes_once(self) -> None:
        guard = ToolGuard(ToolLimits(consequential_tools=frozenset({"create_support_ticket"})))
        args = {"summary": "printer offline", "idempotency_key": "ticket-1"}
        guard.execute(create_support_ticket, args)

        guard.approvals.approve("create_support_ticket", "ticket-1")
        first = guard.execute(create_support_ticket, args)
        second = guard.execute(create_support_ticket, args, call_id="another-call")

        assert first.status == "ok"
        assert second.cached is True
        assert len(ticket_ledger()) == 1


class TestRunBounds:
    def test_cancelled_guard_refuses_before_execution(self) -> None:
        guard = ToolGuard()
        guard.cancel()

        outcome = guard.execute(recorder, {"value": 1})

        assert outcome.stop_reason is StopReason.CANCELLED
        assert _CALLS.get("recorder", 0) == 0

    def test_expired_deadline_refuses_before_execution(self) -> None:
        clock = FakeClock()
        guard = ToolGuard(ToolLimits(deadline_seconds=5), clock=clock)
        clock.advance(6)

        outcome = guard.execute(recorder, {"value": 1})

        assert outcome.stop_reason is StopReason.DEADLINE_EXCEEDED
        assert _CALLS.get("recorder", 0) == 0

    def test_call_budget_refuses_before_execution(self) -> None:
        guard = ToolGuard(ToolLimits(max_total_calls=1, max_calls_per_tool=1))

        assert guard.execute(recorder, {"value": 1}).status == "ok"
        refused = guard.execute(recorder, {"value": 2})

        assert refused.stop_reason is StopReason.CALL_BUDGET_EXCEEDED
        assert _CALLS["recorder"] == 1
        assert guard.budget.snapshot().records[-1].status is CallStatus.REFUSED
        assert guard.budget.snapshot().records[-1].detail == "call_budget_exceeded"
