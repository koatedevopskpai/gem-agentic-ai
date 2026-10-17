"""Tests for the shared execution budget.

The budget is the piece both agent implementations depend on, so its limits
are tested directly rather than only through a model loop. A fake clock makes
the deadline deterministic: no test sleeps to prove a timeout.
"""

import pytest

from gem_agentic_ai.bounds import (
    CallStatus,
    ExecutionBudget,
    StopReason,
    ToolLimits,
    estimate_tokens,
)


class FakeClock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestToolLimits:
    def test_rejects_non_positive_values(self) -> None:
        with pytest.raises(ValueError, match="deadline_seconds"):
            ToolLimits(deadline_seconds=0)

    def test_rejects_per_tool_limit_above_total(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed max_total_calls"):
            ToolLimits(max_total_calls=2, max_calls_per_tool=3)

    def test_accepts_a_coherent_policy(self) -> None:
        limits = ToolLimits(max_total_calls=3, max_calls_per_tool=3)
        assert limits.max_total_calls == 3


class TestEstimateTokens:
    def test_rounds_up(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("a") == 1
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcde") == 2


class TestExecutionBudget:
    def test_total_call_limit_sets_reason(self) -> None:
        budget = ExecutionBudget(ToolLimits(max_total_calls=2, max_calls_per_tool=2))

        assert budget.reserve("a") is None
        assert budget.reserve("b") is None
        assert budget.reserve("a") is StopReason.CALL_BUDGET_EXCEEDED
        assert budget.stop_reason is StopReason.CALL_BUDGET_EXCEEDED
        assert budget.calls == 2

    def test_per_tool_limit_sets_reason(self) -> None:
        budget = ExecutionBudget(ToolLimits(max_total_calls=5, max_calls_per_tool=2))

        assert budget.reserve("a") is None
        assert budget.reserve("a") is None
        assert budget.reserve("a") is StopReason.TOOL_CALL_LIMIT
        # Another tool still has capacity.
        assert budget.reserve("b") is None

    def test_deadline_is_measured_with_the_injected_clock(self) -> None:
        clock = FakeClock()
        budget = ExecutionBudget(ToolLimits(deadline_seconds=10), clock=clock)

        clock.advance(9.9)
        assert budget.reserve("a") is None
        clock.advance(0.2)
        assert budget.reserve("a") is StopReason.DEADLINE_EXCEEDED
        assert clock.now == pytest.approx(10.1)

    def test_cancellation_refuses_new_calls(self) -> None:
        budget = ExecutionBudget(ToolLimits())
        budget.cancel()

        assert budget.cancelled is True
        assert budget.reserve("a") is StopReason.CANCELLED
        assert budget.stop_reason is StopReason.CANCELLED

    def test_token_budget_stops_further_calls(self) -> None:
        budget = ExecutionBudget(ToolLimits(max_observation_tokens=10))

        budget.reserve("a")
        budget.record("a", CallStatus.OK, 1, 0.0, tokens=12)

        assert budget.stop_reason is StopReason.TOKEN_BUDGET_EXCEEDED
        assert budget.reserve("b") is StopReason.TOKEN_BUDGET_EXCEEDED

    def test_first_stop_reason_is_kept(self) -> None:
        budget = ExecutionBudget(ToolLimits(max_consecutive_failures=1))
        budget.note_failure("a")

        assert budget.stop_reason is StopReason.CIRCUIT_OPEN
        budget.cancel()
        # Cancellation is recorded, but it does not rewrite history.
        assert budget.stop_reason is StopReason.CIRCUIT_OPEN

    def test_circuit_opens_after_consecutive_failures(self) -> None:
        budget = ExecutionBudget(ToolLimits(max_consecutive_failures=2))

        budget.note_failure("a")
        assert budget.is_circuit_open("a") is False
        budget.note_failure("a")
        assert budget.is_circuit_open("a") is True
        assert budget.reserve("a") is StopReason.CIRCUIT_OPEN

    def test_success_resets_the_failure_count(self) -> None:
        budget = ExecutionBudget(ToolLimits(max_consecutive_failures=2))

        budget.note_failure("a")
        budget.note_success("a")
        budget.note_failure("a")

        assert budget.is_circuit_open("a") is False

    def test_snapshot_is_a_copy(self) -> None:
        budget = ExecutionBudget(ToolLimits(max_total_calls=3, max_calls_per_tool=3))
        budget.reserve("a")
        budget.record("a", CallStatus.OK, 1, 0.0, tokens=1)

        snapshot = budget.snapshot()
        budget.reserve("a")

        assert snapshot.calls == 1
        assert dict(snapshot.per_tool) == {"a": 1}
        assert snapshot.observation_tokens == 1
        assert len(snapshot.records) == 1

    def test_reset_starts_a_new_run(self) -> None:
        budget = ExecutionBudget(ToolLimits(max_total_calls=1, max_calls_per_tool=1))
        budget.reserve("a")
        assert budget.reserve("b") is StopReason.CALL_BUDGET_EXCEEDED

        budget.reset()

        assert budget.calls == 0
        assert budget.stop_reason is None
        assert budget.reserve("a") is None
