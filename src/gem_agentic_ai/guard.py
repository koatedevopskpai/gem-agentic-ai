"""Bounded tool execution for both agent implementations.

A01 pinned a real difference: the manual ReAct loop caught an unexpected tool
exception and continued, while LangGraph's tool node re-raised it and ended
the run. Rather than patch one path, this module puts every tool call through
one guard. Both implementations therefore share validation, timeouts, budgets,
retry rules, idempotency and approval state.

FAIL-CLOSED RULES
  * A tool without a Pydantic ``args_schema`` is refused. An unvalidated call
    is not executed.
  * Invalid arguments are refused before the tool runs.
  * A tool outside the allowlist is refused.
  * A consequential tool cannot run until its action key has been approved.
  * A timed-out consequential tool is never retried automatically, because a
    partial side effect cannot be undone by retrying.

ERROR CLASSIFICATION
  ``RetryableToolError`` (and its subclass ``ToolTimeoutError``) are retried
  up to ``max_attempts_per_call``. Every other exception is treated as
  non-retryable: retrying an unknown failure risks repeating a side effect.
  Tool authors opt in to retries explicitly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, ValidationError

from gem_agentic_ai.bounds import (
    CallStatus,
    ExecutionBudget,
    StopReason,
    ToolLimits,
    estimate_tokens,
)
from gem_agentic_ai.idempotency import IdempotencyStore, InMemoryIdempotencyStore

logger = logging.getLogger(__name__)


class RetryableToolError(Exception):
    """Raise from a tool when the failure is transient and retry is safe.

    Only raise this when repeating the call cannot duplicate a side effect.
    """


class ToolTimeoutError(RetryableToolError):
    """The guard stopped waiting for a tool call."""


@dataclass(frozen=True)
class ToolOutcome:
    """The result of one requested tool call.

    ``content`` is what the model sees. ``status`` maps onto
    ``ToolMessage.status``. ``stop_reason`` is set when the budget refused the
    call.
    """

    content: str
    status: Literal["ok", "error"]
    error_type: str | None = None
    stop_reason: StopReason | None = None
    cached: bool = False


def action_key(tool_name: str, args: Mapping[str, Any], call_id: str | None = None) -> str:
    """Derive the idempotency and approval key for one call.

    Precedence:
      1. an explicit ``idempotency_key`` argument, which is the only key that
         survives a model generating a fresh call id for a retried action;
      2. the model's call id, when the caller has one;
      3. a stable hash of the arguments.

    Rule 3 is a convenience for read-only tools. A side-effecting tool should
    require an explicit key, because two legitimate calls with identical
    arguments would otherwise collapse into one.
    """
    explicit = args.get("idempotency_key") if isinstance(args, Mapping) else None
    if explicit:
        return str(explicit)
    if call_id:
        return f"call:{call_id}"
    payload = json.dumps(dict(args), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(f"{tool_name}:{payload}".encode()).hexdigest()[:16]


class ApprovalStore:
    """In-process approval state for consequential tools.

    This is deliberately small: a request is recorded when a call is refused,
    and a caller (a CLI prompt, a reviewer, a test) approves the action key.
    The store is not a durable audit log; A02 owns durable state. Its purpose
    is to make "no side effect without an explicit decision" enforceable.
    """

    def __init__(self) -> None:
        self._approved: set[tuple[str, str]] = set()
        self._pending: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def request(self, tool: str, key: str) -> None:
        with self._lock:
            if (tool, key) not in self._approved:
                self._pending.add((tool, key))

    def approve(self, tool: str, key: str) -> None:
        with self._lock:
            self._approved.add((tool, key))
            self._pending.discard((tool, key))

    def is_approved(self, tool: str, key: str) -> bool:
        with self._lock:
            return (tool, key) in self._approved

    @property
    def pending(self) -> frozenset[tuple[str, str]]:
        with self._lock:
            return frozenset(self._pending)


class ToolGuard:
    """One policy layer shared by the manual loop and the LangGraph agent.

    A guard bounds one run. ``reset()`` starts a new run with the same limits;
    the idempotency store and approval decisions deliberately survive it,
    because they describe completed actions rather than run state.
    """

    def __init__(
        self,
        limits: ToolLimits | None = None,
        *,
        approvals: ApprovalStore | None = None,
        idempotency: IdempotencyStore | None = None,
        allowed_tools: Iterable[str] | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.limits = limits or ToolLimits()
        self.budget = ExecutionBudget(self.limits, clock=clock)
        self.approvals = approvals or ApprovalStore()
        self.idempotency: IdempotencyStore = idempotency or InMemoryIdempotencyStore()
        self._allowed_tools = frozenset(allowed_tools) if allowed_tools is not None else None
        self._clock = clock

    @property
    def allowed_tools(self) -> frozenset[str] | None:
        return self._allowed_tools

    def restrict_to(self, names: Iterable[str]) -> None:
        """Set the allowlist. Called by both agent builders from their registry."""
        self._allowed_tools = frozenset(names)

    def reset(self) -> None:
        self.budget.reset()

    def cancel(self) -> None:
        self.budget.cancel()

    @property
    def cancelled(self) -> bool:
        return self.budget.cancelled

    def wrap(self, tool: BaseTool) -> GuardedTool:
        """Return a tool that the LangGraph tool node can call."""
        return GuardedTool(
            wrapped=tool,
            guard=self,
            name=tool.name,
            description=tool.description or tool.name,
            args_schema=tool.args_schema,
        )

    # ── outcomes ─────────────────────────────────────────────────

    def _refused(self, name: str, reason: StopReason) -> ToolOutcome:
        logger.warning("Tool call to %s refused: %s", name, reason.value)
        # Refusals are audited too. A refusal consumes no call slot, but a
        # caller inspecting the run should still see that something was asked
        # for and declined.
        self.budget.record(name, CallStatus.REFUSED, 0, 0.0, detail=reason.value)
        return ToolOutcome(
            content=f"Error: tool call refused — {reason.value}",
            status="error",
            error_type="budget",
            stop_reason=reason,
        )

    def _invalid(self, name: str, detail: str, error_type: str) -> ToolOutcome:
        logger.warning("Tool call to %s rejected: %s", name, detail)
        self.budget.note_failure(name)
        self.budget.record(name, CallStatus.REFUSED, 0, 0.0, detail=detail)
        return ToolOutcome(
            content=f"Error: {detail}",
            status="error",
            error_type=error_type,
        )

    def reject(self, tool_name: str, *, detail: str) -> ToolOutcome:
        """Refuse a call that never reached a registered tool.

        The manual loop uses this for an unknown tool name, so the refusal is
        still counted and recorded rather than disappearing from the audit.
        """
        reason = self.budget.reserve(tool_name)
        if reason is not None:
            return self._refused(tool_name, reason)
        return self._invalid(tool_name, detail, "unknown_tool")

    # ── execution ────────────────────────────────────────────────

    def execute(
        self,
        tool: BaseTool,
        args: Mapping[str, Any],
        *,
        call_id: str | None = None,
    ) -> ToolOutcome:
        """Validate, authorise and execute one tool call within the budget."""
        name = getattr(tool, "name", type(tool).__name__)
        started = self._clock()

        if self._allowed_tools is not None and name not in self._allowed_tools:
            return self._invalid(
                name,
                f"tool '{name}' is not on the allowlist",
                "not_allowed",
            )

        reason = self.budget.reserve(name)
        if reason is not None:
            return self._refused(name, reason)

        if not isinstance(args, Mapping):
            return self._invalid(
                name,
                f"arguments for '{name}' must be a mapping",
                "invalid_arguments",
            )

        try:
            argument_bytes = len(json.dumps(dict(args), default=str).encode("utf-8"))
        except (TypeError, ValueError):
            argument_bytes = len(repr(args).encode("utf-8"))
        if argument_bytes > self.limits.max_argument_bytes:
            return self._invalid(
                name,
                f"arguments for '{name}' exceed {self.limits.max_argument_bytes} bytes",
                "arguments_too_large",
            )

        schema = getattr(tool, "args_schema", None)
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return self._invalid(
                name,
                (
                    f"tool '{name}' declares no Pydantic argument schema; "
                    "refusing to execute unvalidated"
                ),
                "unvalidated_tool",
            )
        try:
            validated = schema.model_validate(dict(args))
        except ValidationError as exc:
            return self._invalid(
                name,
                f"invalid arguments for '{name}': {exc.errors(include_url=False)}",
                "invalid_arguments",
            )

        key = action_key(name, args, call_id)

        cached = self.idempotency.get(name, key)
        if cached is not None:
            self.budget.note_success(name)
            self.budget.record(
                name,
                CallStatus.CACHED,
                0,
                self._clock() - started,
                detail=f"key={key}",
            )
            return ToolOutcome(content=cached, status="ok", cached=True)

        if name in self.limits.consequential_tools and not self.approvals.is_approved(name, key):
            self.approvals.request(name, key)
            self.budget.record(
                name,
                CallStatus.APPROVAL_REQUIRED,
                0,
                self._clock() - started,
                detail=f"key={key}",
            )
            return ToolOutcome(
                content=(
                    f"Error: approval required for '{name}' (action key '{key}'); "
                    "no side effect was executed"
                ),
                status="error",
                error_type="approval_required",
            )

        return self._execute_with_retries(
            tool=tool,
            name=name,
            args=validated.model_dump(),
            key=key,
            started=started,
        )

    def _execute_with_retries(
        self,
        *,
        tool: BaseTool,
        name: str,
        args: dict[str, Any],
        key: str,
        started: float,
    ) -> ToolOutcome:
        consequential = name in self.limits.consequential_tools
        attempts = 0
        last_error: Exception | None = None
        timed_out = False

        while attempts < self.limits.max_attempts_per_call:
            attempts += 1
            remaining = self.budget.remaining_seconds()
            if remaining <= 0:
                last_error = ToolTimeoutError("run deadline reached before the call started")
                timed_out = True
                break
            timeout = min(self.limits.per_call_timeout_seconds, remaining)

            try:
                content = str(self._invoke_with_timeout(tool, args, timeout))
            except ToolTimeoutError as exc:
                last_error = exc
                timed_out = True
                # A partially executed side effect cannot be undone by a retry.
                if consequential or attempts >= self.limits.max_attempts_per_call:
                    break
                time.sleep(min(0.05 * attempts, self.budget.remaining_seconds()))
                continue
            except RetryableToolError as exc:
                last_error = exc
                if attempts >= self.limits.max_attempts_per_call:
                    break
                time.sleep(min(0.05 * attempts, self.budget.remaining_seconds()))
                continue
            except Exception as exc:
                # Unknown failures are not retried: retrying could repeat a
                # side effect whose first attempt partially succeeded.
                last_error = exc
                break

            duration = self._clock() - started
            self.budget.note_success(name)
            self.budget.record(
                name,
                CallStatus.OK,
                attempts,
                duration,
                detail=f"key={key}",
                tokens=estimate_tokens(content),
            )
            self.idempotency.put(name, key, content)
            return ToolOutcome(content=content, status="ok")

        duration = self._clock() - started
        assert last_error is not None  # loop always sets it on exit without success
        error_type = type(last_error).__name__
        status = CallStatus.TIMEOUT if timed_out else CallStatus.ERROR
        detail = f"{error_type}: {last_error}"
        logger.warning("Tool %s failed after %d attempt(s): %s", name, attempts, detail)
        self.budget.note_failure(name)
        self.budget.record(name, status, attempts, duration, detail=detail)
        return ToolOutcome(
            content=f"Error during tool execution: {error_type}: {last_error}",
            status="error",
            error_type=error_type,
        )

    def _invoke_with_timeout(self, tool: BaseTool, args: dict[str, Any], timeout: float) -> Any:
        """Run a synchronous tool call, waiting at most ``timeout`` seconds.

        The call runs in a daemon thread. If it overruns, the guard stops
        waiting and the run continues, but the thread is not killed: Python
        offers no portable way to interrupt it. A non-cooperative tool can
        therefore keep running in the background after its result is
        discarded. True isolation needs a separate process, which is K01's
        scope, not this task's.
        """
        result: list[Any] = []
        error: list[Exception] = []
        done = threading.Event()

        def target() -> None:
            try:
                result.append(tool.invoke(args))
            except Exception as exc:
                error.append(exc)
            finally:
                done.set()

        thread = threading.Thread(
            target=target,
            name=f"tool-{getattr(tool, 'name', 'unknown')}",
            daemon=True,
        )
        thread.start()
        if not done.wait(timeout):
            raise ToolTimeoutError(
                f"tool '{getattr(tool, 'name', 'unknown')}' exceeded {timeout:.2f}s"
            )
        if error:
            raise error[0]
        return result[0]


class GuardedTool(BaseTool):
    """A ``BaseTool`` that delegates to ``ToolGuard``.

    LangGraph's tool node receives the finished ``ToolMessage`` and passes it
    through unchanged, so a refused or failed call is visible with
    ``status="error"`` instead of appearing as a successful observation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped: BaseTool
    guard: ToolGuard

    def _execute(self, args: Mapping[str, Any], call_id: str | None) -> ToolOutcome:
        return self.guard.execute(self.wrapped, args, call_id=call_id)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if isinstance(input, Mapping) and input.get("type") == "tool_call":
            args = input.get("args") or {}
            call_id = input.get("id")
        else:
            args = input
            call_id = None
        outcome = self._execute(args, call_id)
        if call_id is None:
            return outcome.content
        return ToolMessage(
            content=outcome.content,
            name=self.name,
            tool_call_id=call_id,
            # ToolMessage uses "success"/"error"; ToolOutcome uses
            # "ok"/"error" because it is also used outside LangChain.
            status="success" if outcome.status == "ok" else "error",
        )

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.invoke, input, config, **kwargs)

    def _run(self, **kwargs: Any) -> str:
        return self._execute(kwargs, None).content

    async def _arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self._run, **kwargs)
