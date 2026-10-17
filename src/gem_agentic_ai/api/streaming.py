"""One streaming agent turn, with a deadline and cancellation that reaches work.

WHAT THIS LAYER GUARANTEES
  * Tokens are forwarded as the model produces them, not collected first.
    ``agent.astream(..., stream_mode="messages")`` yields one chunk per model
    chunk; LangGraph gives the same shape for a real provider stream.
  * A turn has a deadline. When it expires the run is cancelled, not merely
    abandoned, and the client receives an ``error`` event with code
    ``timeout``.
  * Cancelling the consumer cancels the producer: cancellation is delivered
    into the task iterating ``agent.astream``, which LangGraph treats as
    cancelling the run, and into the model's ``astream`` (and tool
    coroutines), so work stops instead of continuing with nobody listening.
    The run itself executes on a dedicated event loop (``ThreadedTurn``) so
    that an ASGI server's repeated request-scope cancellation cannot
    interrupt that cleanup; see the class docstring for the failure this
    prevents.
  * A turn that did not finish is rolled back at the message level. LangGraph
    checkpoints the user message before the model runs, so an interrupted
    turn without cleanup would leave a question with no answer in the thread
    and the next turn would see two consecutive user messages. The dangling
    message is removed only when it is still the last one; a turn whose answer
    was already checkpointed is left alone.

Timing is measured in-process with ``time.perf_counter``: ``first_token_ms``
is the wait before the first chunk reaches this layer, and ``total_ms`` is the
turn's wall time. Those measure the API and the model together, not the
network to the client.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage, RemoveMessage

logger = logging.getLogger(__name__)

DEFAULT_DEADLINE_SECONDS = 30.0
MIN_DEADLINE_SECONDS = 0.05
MAX_DEADLINE_SECONDS = 300.0
MAX_MESSAGE_CHARS = 8000

TurnKind = Literal["token", "done", "error"]


@dataclass(frozen=True)
class TurnTiming:
    first_token_ms: float | None
    total_ms: float
    tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "first_token_ms": None
            if self.first_token_ms is None
            else round(self.first_token_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "tokens": self.tokens,
        }


@dataclass(frozen=True)
class TurnEvent:
    """One SSE event: a token, a completed answer, or a failure."""

    kind: TurnKind
    text: str = ""
    code: str = ""
    message: str = ""
    timing: TurnTiming | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def resolve_deadline(value: float | None) -> float:
    """Validate a per-turn deadline, falling back to the default."""
    if value is None:
        return DEFAULT_DEADLINE_SECONDS
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("deadline_seconds must be a number")
    if value < MIN_DEADLINE_SECONDS or value > MAX_DEADLINE_SECONDS:
        raise ValueError(
            f"deadline_seconds must be between {MIN_DEADLINE_SECONDS} "
            f"and {MAX_DEADLINE_SECONDS}"
        )
    return float(value)


def validate_message(text: str) -> str:
    """Validate one user message. Empty or oversized input is refused."""
    if not isinstance(text, str):
        raise ValueError("message must be a string")
    text = text.strip()
    if not text:
        raise ValueError("message must not be empty")
    if len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS} characters")
    return text


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                pieces.append(block["text"])
        return "".join(pieces)
    return ""


def _is_model_chunk(chunk: Any, metadata: Any) -> bool:
    """Ignore chunks from tool nodes; the client sees assistant text only."""
    node = (metadata or {}).get("langgraph_node")
    return node == "model" and _chunk_text(chunk) != ""


def _discard_unanswered_turn(agent: Any, config: Any, text: str) -> bool:
    """Remove this turn's user message if it is still unanswered.

    Synchronous on purpose: it is executed in a worker thread so that a
    cancellation of the awaiting task cannot interrupt the write halfway.
    Returns ``True`` when a message was removed. The content check makes the
    rollback conservative: if the model already produced an answer (so the
    last message is the assistant's), or another writer changed the thread,
    nothing is touched.
    """
    state = agent.get_state(config)
    messages = getattr(state, "values", {}).get("messages", []) if state else []
    if not messages:
        return False
    last = messages[-1]
    if isinstance(last, HumanMessage) and _chunk_text(last) == text:
        agent.update_state(config, {"messages": [RemoveMessage(id=last.id)]})
        return True
    return False


async def _rollback(agent: Any, config: Any, text: str) -> None:
    """Remove an unanswered message; never raise on failure.

    This runs in the ``finally`` of a cancelled turn. The SQLite work runs in
    a thread: once started it completes even if the awaiting task is
    cancelled again, so a disconnect cannot leave an orphaned question behind
    the way a partially-awaited checkpoint write could.
    """
    cleanup = asyncio.ensure_future(
        asyncio.to_thread(_discard_unanswered_turn, agent, config, text)
    )
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        logger.warning("rollback of an unfinished turn was interrupted")
    except Exception:
        logger.exception("failed to roll back an unfinished turn")


async def stream_turn(
    agent: Any,
    config: Any,
    text: str,
    *,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> AsyncIterator[TurnEvent]:
    """Run one turn, yielding tokens and then ``done`` or ``error``.

    The caller cancels the consuming task (or closes the generator) to stop
    the turn; cancellation reaches the model and tools, and the unanswered
    user message is rolled back. A deadline raises the same path and is
    reported as a ``timeout`` error event.
    """
    started = time.perf_counter()
    first_token_at: float | None = None
    parts: list[str] = []
    completed = False
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def drive() -> None:
        """Consume the LangGraph stream in its own task.

        WHY A TASK. Cancelling the *consumer* of ``agent.astream`` only closes
        the async generator that wraps it. LangGraph's Pregel loop runs the
        graph in its own task, and a ``GeneratorExit`` from closing the outer
        generator is not treated as a cancellation of that work: the model
        stream keeps producing with nobody listening. Delivering
        ``CancelledError`` from inside a task that is iterating the stream does
        stop the run (verified against the model double, which records that its
        stream was cancelled rather than merely closed).
        """
        stream = agent.astream(
            {"messages": [{"role": "user", "content": text}]},
            config,
            stream_mode="messages",
        )
        try:
            async for chunk, metadata in stream:
                if _is_model_chunk(chunk, metadata):
                    await queue.put(("token", _chunk_text(chunk)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # reported to the client by the caller
            await queue.put(("error", exc))
        finally:
            # Cancellation may have been delivered while this task was
            # suspended inside ``__anext__``; closing explicitly is what
            # unwinds the agent stream and stops in-flight node tasks.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stream.aclose()
            await queue.put(("end", None))

    driver = asyncio.create_task(drive())
    try:
        async with asyncio.timeout(deadline_seconds):
            while True:
                kind, value = await queue.get()
                if kind == "end":
                    break
                if kind == "error":
                    raise value
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(value)
                yield TurnEvent("token", text=value)
        completed = True
    except TimeoutError:
        yield TurnEvent(
            "error",
            code="timeout",
            message=f"the turn exceeded its {deadline_seconds:g}s deadline and was cancelled",
            timing=TurnTiming(
                None if first_token_at is None else (first_token_at - started) * 1000,
                (time.perf_counter() - started) * 1000,
                len(parts),
            ),
        )
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("agent turn failed")
        yield TurnEvent(
            "error",
            code="agent_error",
            message=f"{type(exc).__name__}: {exc}",
            timing=TurnTiming(
                None if first_token_at is None else (first_token_at - started) * 1000,
                (time.perf_counter() - started) * 1000,
                len(parts),
            ),
        )
        return
    finally:
        # Stop the run before rolling back: the rollback edits the thread, and
        # it must not race a model that is still writing checkpoints.
        if not driver.done():
            driver.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await driver
        if not completed:
            with contextlib.suppress(Exception):
                await _rollback(agent, config, text)
    yield TurnEvent(
        "done",
        text="".join(parts),
        timing=TurnTiming(
            None if first_token_at is None else (first_token_at - started) * 1000,
            (time.perf_counter() - started) * 1000,
            len(parts),
        ),
    )


class ThreadedTurn:
    """Own one turn on a dedicated event loop in a worker thread.

    WHY ITS OWN LOOP. An ASGI server runs each response inside a cancel scope
    that is cancelled when the client disconnects, and the server repeats
    that cancellation on every event-loop iteration until the response
    finishes. Every await in the request's task tree is interrupted again as
    soon as it resumes, including the awaits inside LangGraph's shutdown,
    which is where the tasks running the model get cancelled. In practice the
    run was never cancelled: the socket closed, the answer was rolled back,
    and the model kept producing tokens into an empty queue.

    A turn on this dedicated loop receives exactly one cancellation, from
    :meth:`cancel`, and its cleanup runs to completion regardless of what the
    HTTP request task is doing. Events are handed to the request loop with
    ``call_soon_threadsafe``; the request loop only forwards them.
    """

    def __init__(
        self,
        agent: Any,
        config: Any,
        text: str,
        *,
        deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        self._agent = agent
        self._config = config
        self._text = text
        self._deadline_seconds = deadline_seconds
        self._on_finish = on_finish
        self._request_loop: asyncio.AbstractEventLoop | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._cancel_requested = False
        self._queue: asyncio.Queue[TurnEvent | None] | None = None
        self._thread = threading.Thread(
            target=self._run, name="agent-turn", daemon=True
        )

    def start(self, request_loop: asyncio.AbstractEventLoop) -> ThreadedTurn:
        """Start the worker. The request loop receives the events."""
        self._request_loop = request_loop
        self._queue = asyncio.Queue()
        self._thread.start()
        return self

    @property
    def events(self) -> asyncio.Queue[TurnEvent | None]:
        """Events for the request loop, ending with ``None``."""
        assert self._queue is not None, "start() the turn before reading events"
        return self._queue

    def cancel(self) -> None:
        """Ask the turn to stop. Safe from the request loop at any time."""
        self._cancel_requested = True
        loop, task = self._worker_loop, self._task
        if loop is None or loop.is_closed():
            # The worker has not started yet; it checks the flag first.
            return
        # ``_task`` is assigned before ``_worker_loop``, so it is set here.
        if task is not None:
            loop.call_soon_threadsafe(task.cancel)

    def _publish(self, item: TurnEvent | None) -> None:
        loop = self._request_loop
        if loop is None or loop.is_closed():
            return
        queue = self._queue
        assert queue is not None
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def _notify_finished(self) -> None:
        """Tell the request loop that cleanup is complete, on its loop.

        Owner after the worker's own sentinel: the HTTP response may have
        ended long before this, and the session must stay claimed until the
        rollback has actually finished or the next turn could read and write
        the thread while this one is still cleaning up.
        """
        callback = self._on_finish
        loop = self._request_loop
        if callback is None or loop is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(callback)

    def _run(self) -> None:
        async def turn() -> None:
            self._task = asyncio.current_task()
            self._worker_loop = asyncio.get_running_loop()
            if self._cancel_requested:
                self._task.cancel()
            try:
                async for event in stream_turn(
                    self._agent,
                    self._config,
                    self._text,
                    deadline_seconds=self._deadline_seconds,
                ):
                    self._publish(event)
            except asyncio.CancelledError:
                pass
            finally:
                self._publish(None)

        try:
            asyncio.run(turn())
        except asyncio.CancelledError:
            logger.debug("turn worker cancelled")
        except Exception:
            logger.exception("turn worker failed")
            self._publish(
                TurnEvent(
                    "error",
                    code="internal_error",
                    message="the turn worker failed; see the server log",
                )
            )
        finally:
            self._notify_finished()


def encode_sse(event: TurnEvent) -> str:
    """Render one event as a Server-Sent Events frame."""
    payload: dict[str, Any] = {}
    if event.kind == "token":
        payload["text"] = event.text
    elif event.kind == "done":
        payload["text"] = event.text
        if event.timing is not None:
            payload["timing"] = event.timing.as_dict()
    else:
        payload["code"] = event.code
        payload["message"] = event.message
        if event.timing is not None:
            payload["timing"] = event.timing.as_dict()
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event.kind}\ndata: {data}\n\n"
