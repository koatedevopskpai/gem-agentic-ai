"""Execution bounds shared by both agent implementations.

WHY a shared budget rather than framework middleware?
  A01 showed that the two agent implementations could disagree about failure
  behaviour. LangChain 1.x offers ``ToolCallLimitMiddleware`` and
  ``ToolRetryMiddleware`` for the graph path, but the manual ReAct loop would
  still need its own logic. Two implementations of the same policy is how the
  divergence appeared in the first place, so both paths share this module.

WHAT IS BOUNDED
  * wall-clock deadline for the run
  * per-call execution timeout
  * total tool calls and calls per tool
  * accumulated observation size, measured in estimated tokens
  * consecutive failures per tool (a per-tool circuit breaker)
  * cancellation requested by the caller

WHAT IS NOT BOUNDED HERE
  * model latency and model spend. The LLM client owns its own timeout; this
    layer never sees the provider's token usage.
  * pre-emptive cancellation of a tool that ignores its deadline. The guard
    stops waiting; it cannot kill a Python thread. See ``docs/A03.md``.

STOP REASONS ARE RECORDED, NOT INFERRED
  The first refusal sets ``ExecutionBudget.stop_reason`` and it is never
  cleared for the rest of the run. A caller can therefore explain why a run
  was cut short even when the model still produced an answer afterwards.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class StopReason(StrEnum):
    """Why a bounded run, or a call within it, was stopped."""

    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CALL_BUDGET_EXCEEDED = "call_budget_exceeded"
    TOOL_CALL_LIMIT = "tool_call_limit"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    CIRCUIT_OPEN = "circuit_open"


class CallStatus(StrEnum):
    """Outcome of one requested tool call, as recorded in the audit trail."""

    OK = "ok"
    CACHED = "cached"
    ERROR = "error"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    APPROVAL_REQUIRED = "approval_required"


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count using ``len(text) // 4``.

    This is deliberately an estimate. Its only job is to stop a run from
    accumulating unbounded text in its context. It is not a billing measure
    and will not match a tokenizer exactly.
    """
    return (len(text) + 3) // 4


@dataclass(frozen=True)
class ToolLimits:
    """Immutable limits for one agent run.

    Values are validated on construction: a nonsensical policy fails before a
    run starts rather than silently disabling a bound.

    Attributes:
        deadline_seconds: Wall-clock budget for the whole run.
        per_call_timeout_seconds: Maximum time the guard waits for one tool
            call. The tool may continue in a background daemon thread if it
            does not cooperate; see ``docs/A03.md``.
        max_total_calls: Maximum number of requested tool calls in the run,
            including invalid and refused calls.
        max_calls_per_tool: Maximum number of requested calls to one tool.
        max_attempts_per_call: Attempts for a retryable failure. Non-retryable
            failures are never retried.
        max_consecutive_failures: Consecutive failures per tool before its
            circuit opens and further calls are refused.
        max_observation_tokens: Accumulated estimated tokens of tool
            observations before further calls are refused.
        max_argument_bytes: Maximum serialised size of a tool call's
            arguments, to bound accidental or malicious payloads.
        consequential_tools: Tool names that require an explicit approval and
            are never automatically retried after a timeout.
    """

    deadline_seconds: float = 30.0
    per_call_timeout_seconds: float = 5.0
    max_total_calls: int = 12
    max_calls_per_tool: int = 4
    max_attempts_per_call: int = 2
    max_consecutive_failures: int = 3
    max_observation_tokens: int = 4000
    max_argument_bytes: int = 4000
    consequential_tools: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        positive = {
            "deadline_seconds": self.deadline_seconds,
            "per_call_timeout_seconds": self.per_call_timeout_seconds,
            "max_total_calls": self.max_total_calls,
            "max_calls_per_tool": self.max_calls_per_tool,
            "max_attempts_per_call": self.max_attempts_per_call,
            "max_consecutive_failures": self.max_consecutive_failures,
            "max_observation_tokens": self.max_observation_tokens,
            "max_argument_bytes": self.max_argument_bytes,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"ToolLimits.{name} must be greater than zero, got {value!r}")
        if self.max_calls_per_tool > self.max_total_calls:
            raise ValueError(
                "ToolLimits.max_calls_per_tool cannot exceed max_total_calls: "
                f"{self.max_calls_per_tool} > {self.max_total_calls}"
            )


@dataclass(frozen=True)
class CallRecord:
    """One line in the run's audit trail."""

    tool: str
    status: CallStatus
    attempts: int
    duration_seconds: float
    detail: str = ""


@dataclass(frozen=True)
class BudgetSnapshot:
    """Immutable view of a run's counters, safe to expose to a caller."""

    stop_reason: StopReason | None
    calls: int
    per_tool: Mapping[str, int]
    observation_tokens: int
    elapsed_seconds: float
    cancelled: bool
    records: tuple[CallRecord, ...]


@dataclass
class ExecutionBudget:
    """Mutable counters for one bounded run.

    The guard is the only intended caller. Public access is limited to
    ``snapshot()`` and ``cancel()`` so a caller cannot reset a budget halfway
    through a run and bypass a bound.
    """

    limits: ToolLimits
    clock: Callable[[], float] = time.monotonic
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _started_at: float = field(init=False, repr=False)
    _stop_reason: StopReason | None = field(default=None, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)
    _per_tool: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _observation_tokens: int = field(default=0, init=False, repr=False)
    _failures: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _circuits: set[str] = field(default_factory=set, init=False, repr=False)
    _records: list[CallRecord] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._started_at = self.clock()

    def reset(self) -> None:
        """Start a new run with the same limits."""
        with self._lock:
            self._started_at = self.clock()
            self._stop_reason = None
            self._cancelled = False
            self._calls = 0
            self._per_tool.clear()
            self._observation_tokens = 0
            self._failures.clear()
            self._circuits.clear()
            self._records.clear()

    def cancel(self) -> None:
        """Request cancellation. New calls are refused from this point."""
        with self._lock:
            self._cancelled = True
            if self._stop_reason is None:
                self._stop_reason = StopReason.CANCELLED

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def stop_reason(self) -> StopReason | None:
        with self._lock:
            return self._stop_reason

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def elapsed_seconds(self) -> float:
        with self._lock:
            return max(0.0, self.clock() - self._started_at)

    def remaining_seconds(self) -> float:
        with self._lock:
            return max(0.0, self.limits.deadline_seconds - (self.clock() - self._started_at))

    def _stop_locked(self, reason: StopReason) -> StopReason:
        if self._stop_reason is None:
            self._stop_reason = reason
        return reason

    def check(self, tool_name: str) -> StopReason | None:
        """Return the reason a call must not start, without consuming a slot."""
        with self._lock:
            if self._cancelled:
                return self._stop_locked(StopReason.CANCELLED)
            if self.clock() - self._started_at >= self.limits.deadline_seconds:
                return self._stop_locked(StopReason.DEADLINE_EXCEEDED)
            if self._calls >= self.limits.max_total_calls:
                return self._stop_locked(StopReason.CALL_BUDGET_EXCEEDED)
            if self._per_tool.get(tool_name, 0) >= self.limits.max_calls_per_tool:
                return self._stop_locked(StopReason.TOOL_CALL_LIMIT)
            if tool_name in self._circuits:
                return self._stop_locked(StopReason.CIRCUIT_OPEN)
            if self._observation_tokens >= self.limits.max_observation_tokens:
                return self._stop_locked(StopReason.TOKEN_BUDGET_EXCEEDED)
            return None

    def reserve(self, tool_name: str) -> StopReason | None:
        """Consume one call slot, or return the reason it was refused."""
        with self._lock:
            reason = self.check(tool_name)
            if reason is not None:
                return reason
            self._calls += 1
            self._per_tool[tool_name] = self._per_tool.get(tool_name, 0) + 1
            return None

    def note_success(self, tool_name: str) -> None:
        with self._lock:
            self._failures[tool_name] = 0

    def note_failure(self, tool_name: str) -> None:
        """Count a failure and open the tool's circuit at the threshold."""
        with self._lock:
            failures = self._failures.get(tool_name, 0) + 1
            self._failures[tool_name] = failures
            if failures >= self.limits.max_consecutive_failures:
                self._circuits.add(tool_name)
                self._stop_locked(StopReason.CIRCUIT_OPEN)

    def is_circuit_open(self, tool_name: str) -> bool:
        with self._lock:
            return tool_name in self._circuits

    def record(
        self,
        tool_name: str,
        status: CallStatus,
        attempts: int,
        duration_seconds: float,
        *,
        detail: str = "",
        tokens: int = 0,
    ) -> None:
        with self._lock:
            self._records.append(
                CallRecord(
                    tool=tool_name,
                    status=status,
                    attempts=attempts,
                    duration_seconds=duration_seconds,
                    detail=detail,
                )
            )
            self._observation_tokens += tokens
            if (
                tokens
                and self._observation_tokens >= self.limits.max_observation_tokens
                and self._stop_reason is None
            ):
                self._stop_reason = StopReason.TOKEN_BUDGET_EXCEEDED

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                stop_reason=self._stop_reason,
                calls=self._calls,
                per_tool=dict(self._per_tool),
                observation_tokens=self._observation_tokens,
                elapsed_seconds=max(0.0, self.clock() - self._started_at),
                cancelled=self._cancelled,
                records=tuple(self._records),
            )


@dataclass(frozen=True)
class AgentRunResult:
    """What an agent run returns when its execution is bounded.

    ``stop_reason`` is the loop's own stopping condition. ``budget`` carries
    the guard's independent record, including the first call it refused, so a
    completed answer does not hide a tool that was cut off along the way.
    """

    answer: str
    stop_reason: StopReason
    budget: BudgetSnapshot

    def __str__(self) -> str:
        return self.answer
