"""Async HTTP API over the agent: sessions, token streaming, cancellation.

DESIGN NOTES
  * **Transport** — Server-Sent Events over HTTP. A turn is one POST whose
    response is an event stream (``token`` frames, then ``done`` or
    ``error``). SSE is one-way, proxy-friendly, and a disconnected client can
    reconnect and rehydrate with a plain GET. WebSockets were rejected: the
    agent does not need client-to-server frames mid-turn, and SSE keeps one
    request per turn, which makes ownership and cancellation easier to reason
    about.
  * **Sessions** — A session is an A02 durable thread. The client receives the
    thread's opaque storage id and never supplies a user id. Every lookup is
    scoped by the authenticated user, and another user's session id yields
    404, the same response as an id that does not exist.
  * **Ownership of work** — Each turn runs on a dedicated event loop in a
    worker thread (``ThreadedTurn``) and the request coroutine only forwards
    its events. A client disconnect cancels that run explicitly. This is not
    incidental: under Starlette/AnyIO the request's cancel scope re-cancels
    every task in the response on each event-loop iteration, which interrupted
    LangGraph's shutdown before it could cancel the model node — the socket
    closed but tokens kept being produced. The tests assert that the model's
    stream was cancelled, not merely that the connection ended.
  * **One turn per session** — a second concurrent POST to the same session
    is rejected with 409 rather than interleaving two writers in one thread.
  * **Agent per turn** — each turn builds its own guard and graph so that two
    sessions never share a budget or an idempotency window. Graph compilation
    is measured in ``evidence/A05.md``; it is milliseconds for this agent.

WHAT THIS IS NOT
  The token map is a demo authenticator, not OIDC. There is no rate limiting,
  no tenancy beyond a user id, no persistence of API keys, and the service
  speaks text only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from gem_agentic_ai.agent.graph import BoundedAgent, build_agent
from gem_agentic_ai.api.auth import AuthError, TokenAuthenticator, parse_tokens
from gem_agentic_ai.api.sessions import SessionNotFoundError, SessionService
from gem_agentic_ai.api.streaming import (
    DEFAULT_DEADLINE_SECONDS,
    ThreadedTurn,
    encode_sse,
    resolve_deadline,
    validate_message,
)
from gem_agentic_ai.persistence import DurableMemory, ThreadRecord

logger = logging.getLogger(__name__)

DISCONNECT_POLL_SECONDS = 0.05
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


class SessionCreate(BaseModel):
    """Body for ``POST /sessions``."""

    label: str | None = None


class MessageRequest(BaseModel):
    """Body for ``POST /sessions/{session_id}/messages``."""

    message: str
    deadline_seconds: float | None = None


def _session_json(record: ThreadRecord) -> dict[str, Any]:
    return {
        "session_id": record.storage_id,
        "label": record.thread_key,
        "created_at": record.created_at,
        "last_used_at": record.last_used_at,
    }


def create_app(
    *,
    llm: BaseChatModel,
    tokens: Mapping[str, str] | str,
    memory: DurableMemory,
    tools: Sequence[BaseTool] = (),
    system_prompt: str | None = None,
    default_deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    disconnect_poll_seconds: float = DISCONNECT_POLL_SECONDS,
) -> FastAPI:
    """Build the ASGI application.

    Args:
        llm: The chat model. Production passes ``ChatOpenAI``; tests pass a
            scripted streaming model so the suite stays offline.
        tokens: Token-to-user map, or the ``token:user,...`` string from
            ``GEM_API_TOKENS``. Empty input raises: anonymous access is not a
            mode.
        memory: The durable store. The caller owns it and closes it.
        tools: Tools the agent may call. Defaults to none, which is enough for
            the streaming path.
        system_prompt: Optional system message for every turn.
        default_deadline_seconds: Deadline applied when a request does not set
            one.
        disconnect_poll_seconds: How often the stream checks for a client
            disconnect while waiting for tokens. Small enough to stop work
            promptly, large enough not to busy-loop.
    """
    if isinstance(tokens, str):
        tokens = parse_tokens(tokens)
    authenticator = TokenAuthenticator(tokens)
    sessions = SessionService(memory)
    active_turns: set[str] = set()
    default_deadline = resolve_deadline(default_deadline_seconds)

    app = FastAPI(
        title="Gem Agentic AI API",
        version="0.1.0",
        description="Streaming conversations over durable, user-scoped sessions.",
    )

    def authenticate(request: Request) -> str:
        try:
            return authenticator.user_for(request.headers.get("authorization"))
        except AuthError as exc:
            raise HTTPException(
                status_code=401,
                detail="unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    def session_or_404(user_id: str, session_id: str) -> ThreadRecord:
        try:
            return sessions.resolve(user_id, session_id)
        except SessionNotFoundError as exc:
            # Deliberately identical for "does not exist" and "belongs to
            # another user": the API must not confirm the existence of a
            # session the caller cannot access.
            raise HTTPException(status_code=404, detail="session not found") from exc

    def build_turn_agent() -> BoundedAgent:
        return build_agent(
            llm,
            list(tools),
            system_prompt=system_prompt,
            checkpointer=memory.checkpointer,
            guard=memory.guard(),
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", status_code=201)
    async def create_session(
        body: SessionCreate, user_id: str = Depends(authenticate)
    ) -> dict[str, Any]:
        try:
            record = sessions.create(user_id, body.label)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _session_json(record)

    @app.get("/sessions")
    async def list_sessions(user_id: str = Depends(authenticate)) -> dict[str, Any]:
        return {"sessions": [_session_json(r) for r in sessions.list(user_id)]}

    @app.get("/sessions/{session_id}")
    async def get_session(
        session_id: str, user_id: str = Depends(authenticate)
    ) -> dict[str, Any]:
        record = session_or_404(user_id, session_id)
        messages, awaiting = sessions.history(user_id, session_id)
        return {
            **_session_json(record),
            "messages": [message.as_dict() for message in messages],
            "awaiting_reply": awaiting,
        }

    @app.delete("/sessions/{session_id}", status_code=204)
    async def delete_session(
        session_id: str, user_id: str = Depends(authenticate)
    ) -> Response:
        session_or_404(user_id, session_id)
        sessions.delete(user_id, session_id)
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/messages")
    async def send_message(
        session_id: str,
        body: MessageRequest,
        request: Request,
        user_id: str = Depends(authenticate),
    ) -> StreamingResponse:
        session_or_404(user_id, session_id)
        try:
            text = validate_message(body.message)
            deadline = resolve_deadline(
                body.deadline_seconds
                if body.deadline_seconds is not None
                else default_deadline
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        agent = build_turn_agent()
        config = sessions.config(user_id, session_id)
        if session_id in active_turns:
            raise HTTPException(
                status_code=409, detail="a turn is already running for this session"
            )
        # Claim before returning the response so two POSTs cannot both pass
        # the check. No await separates the check from the add, so on one
        # event loop the pair is atomic.
        active_turns.add(session_id)

        def release_claim() -> None:
            # Called by the worker after the rollback, not when the response
            # ends: the response can end while cleanup is still running, and
            # releasing then would let the next turn write the thread this
            # one is still rolling back.
            active_turns.discard(session_id)

        async def events() -> AsyncIterator[str]:
            async for frame in _stream_frames(
                request,
                agent,
                config,
                text,
                deadline,
                disconnect_poll_seconds,
                on_finish=release_claim,
            ):
                yield frame

        return StreamingResponse(
            events(), media_type="text/event-stream", headers=dict(_SSE_HEADERS)
        )

    return app


async def _stream_frames(
    request: Request,
    agent: BoundedAgent,
    config: dict[str, Any],
    text: str,
    deadline: float,
    poll_seconds: float,
    *,
    on_finish: Callable[[], None] | None = None,
) -> AsyncIterator[str]:
    """Drive one turn, checking for a client disconnect while waiting.

    The turn runs on its own event loop (see ``ThreadedTurn``); this
    generator only forwards its events. On a disconnect it asks that turn to
    cancel, which stops the model rather than merely closing the socket.
    ``on_finish`` is called on the request loop once the worker has finished
    its cleanup, whether or not the response is still open.
    """
    turn = ThreadedTurn(
        agent, config, text, deadline_seconds=deadline, on_finish=on_finish
    )
    try:
        turn.start(asyncio.get_running_loop())
    except Exception:
        # The worker never ran, so it cannot release; do it here.
        if on_finish is not None:
            on_finish()
        raise
    queue = turn.events
    finished = False
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=poll_seconds)
            except TimeoutError:
                if await request.is_disconnected():
                    break
                continue
            if event is None:
                # The worker has finished, including any rollback.
                finished = True
                break
            if await request.is_disconnected():
                break
            yield encode_sse(event)
            if event.kind in ("done", "error"):
                finished = True
                break
    finally:
        if not finished:
            turn.cancel()
        else:
            # Wait for the worker's closing sentinel so a rollback that
            # follows an error has completed before the response ends. The
            # worker never waits on this await, so an interruption here only
            # makes the client's next read race, not the cleanup.
            sentinel_seen = False
            with contextlib.suppress(asyncio.CancelledError, Exception):
                sentinel_seen = (await asyncio.wait_for(queue.get(), timeout=5)) is None
            if sentinel_seen and on_finish is not None:
                # Cleanup is done, so release the session before the response
                # ends rather than waiting for the worker's own callback:
                # that callback is queued behind the sentinel but runs only
                # after ``asyncio.run`` tears its loop down, which can be
                # several milliseconds later on Windows. Without this, a
                # client's next turn on the same session would see a 409.
                on_finish()
    # A turn that ended without a sentinel (disconnect) is released by the
    # worker once its rollback finishes; see ``ThreadedTurn._notify_finished``.
