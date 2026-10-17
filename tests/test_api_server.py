"""End-to-end API tests: a real HTTP server, a real socket, real disconnects.

The unit tests cover the streaming service directly. These tests run the app
under uvicorn on a loopback socket and use httpx, because the properties that
matter here only exist over a real connection: an SSE body that arrives
incrementally, a client that closes mid-stream, and a second request that
must be refused while the first turn is running.
"""

import asyncio
import contextlib
import json
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI

from gem_agentic_ai.api.server import create_app
from gem_agentic_ai.persistence import DurableMemory
from tests.fakes import StreamingChatModel

TOKENS = {"t-alice": "alice", "t-bob": "bob"}
ALICE = {"Authorization": "Bearer t-alice"}
BOB = {"Authorization": "Bearer t-bob"}


@contextlib.contextmanager
def _serve(app: FastAPI) -> Iterator[str]:
    """Run uvicorn on an ephemeral loopback port in a background thread."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level="warning", access_log=False, lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        with contextlib.suppress(OSError):
            sock.close()


@contextlib.contextmanager
def live_server(
    tmp_path: Path,
    turns: list[list[str]],
    *,
    delay: float = 0.0,
    default_deadline: float = 30.0,
) -> Iterator[tuple[str, StreamingChatModel]]:
    memory = DurableMemory(tmp_path / "memory")
    model = StreamingChatModel(turns=turns, delay=delay)
    app = create_app(
        llm=model,
        tokens=TOKENS,
        memory=memory,
        default_deadline_seconds=default_deadline,
    )
    try:
        with _serve(app) as url:
            yield url, model
    finally:
        memory.close()


async def _events(response: httpx.Response) -> AsyncIterator[dict]:
    current: dict = {}
    async for line in response.aiter_lines():
        if line.startswith("event: "):
            current["event"] = line[len("event: "):]
        elif line.startswith("data: "):
            current["data"] = json.loads(line[len("data: "):])
        elif line == "" and current:
            yield current
            current = {}


async def _read_events(response: httpx.Response) -> list[dict]:
    return [event async for event in _events(response)]


async def _create_session(client: httpx.AsyncClient, headers: dict = ALICE) -> dict:
    response = await client.post("/sessions", json={}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


async def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


def test_requests_without_a_valid_token_are_rejected(tmp_path: Path) -> None:
    with live_server(tmp_path, turns=[]) as (url, _):
        async def scenario():
            async with httpx.AsyncClient(base_url=url) as client:
                missing = await client.get("/sessions")
                wrong = await client.get(
                    "/sessions", headers={"Authorization": "Bearer nope"}
                )
                malformed = await client.get(
                    "/sessions", headers={"Authorization": "t-alice"}
                )
            return missing, wrong, malformed

        missing, wrong, malformed = asyncio.run(scenario())
        assert missing.status_code == 401
        assert wrong.status_code == 401
        assert malformed.status_code == 401
        assert missing.headers["www-authenticate"] == "Bearer"


def test_a_session_cannot_be_read_by_another_user(tmp_path: Path) -> None:
    with live_server(tmp_path, turns=[["secret answer"]]) as (url, _):
        async def scenario():
            async with httpx.AsyncClient(base_url=url) as client:
                session = await _create_session(client)
                sid = session["session_id"]
                get_other = await client.get(f"/sessions/{sid}", headers=BOB)
                delete_other = await client.delete(f"/sessions/{sid}", headers=BOB)
                post_other = await client.post(
                    f"/sessions/{sid}/messages",
                    json={"message": "steal"},
                    headers=BOB,
                )
                listed = await client.get("/sessions", headers=BOB)
                own = await client.get(f"/sessions/{sid}", headers=ALICE)
            return get_other, delete_other, post_other, listed, own

        get_other, delete_other, post_other, listed, own = asyncio.run(scenario())
        assert get_other.status_code == 404
        assert delete_other.status_code == 404
        assert post_other.status_code == 404
        assert listed.json()["sessions"] == []
        assert own.status_code == 200
        assert own.json()["messages"] == []


def test_tokens_arrive_incrementally_with_timings(tmp_path: Path) -> None:
    with live_server(tmp_path, turns=[["Hello", ", ", "world"]]) as (url, model):
        async def scenario():
            async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                session = await _create_session(client)
                sid = session["session_id"]
                async with client.stream(
                    "POST",
                    f"/sessions/{sid}/messages",
                    json={"message": "greet me"},
                    headers=ALICE,
                ) as response:
                    assert response.status_code == 200
                    assert response.headers["content-type"].startswith("text/event-stream")
                    events = await _read_events(response)
                history = await client.get(f"/sessions/{sid}", headers=ALICE)
            return events, history

        events, history = asyncio.run(scenario())
        tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
        assert tokens == ["Hello", ", ", "world"]
        assert events[-1]["event"] == "done"
        assert events[-1]["data"]["text"] == "Hello, world"
        timing = events[-1]["data"]["timing"]
        assert timing["tokens"] == 3
        assert timing["first_token_ms"] is not None
        assert timing["total_ms"] >= timing["first_token_ms"]

        body = history.json()
        assert body["awaiting_reply"] is False
        assert [(m["role"], m["content"]) for m in body["messages"]] == [
            ("user", "greet me"),
            ("assistant", "Hello, world"),
        ]


def test_a_client_disconnect_cancels_the_run_and_cleans_up(tmp_path: Path) -> None:
    long_turn = [f"t{i} " for i in range(100)]
    with live_server(tmp_path, turns=[long_turn, ["recovered"]], delay=0.02) as (url, model):
        async def scenario():
            async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                session = await _create_session(client)
                sid = session["session_id"]
                received = 0
                async with client.stream(
                    "POST",
                    f"/sessions/{sid}/messages",
                    json={"message": "abandon me"},
                    headers=ALICE,
                ) as response:
                    async for event in _events(response):
                        if event["event"] == "token":
                            received += 1
                        if received >= 2:
                            break
                cancelled = await _wait_until(lambda: model.cancelled)
                emitted_at_cancel = model.emitted
                await asyncio.sleep(0.2)
                emitted_later = model.emitted
                history = (await client.get(f"/sessions/{sid}", headers=ALICE)).json()

                # The session is still usable: a new turn completes.
                async with client.stream(
                    "POST",
                    f"/sessions/{sid}/messages",
                    json={"message": "try again"},
                    headers=ALICE,
                ) as response:
                    events = await _read_events(response)
            return received, cancelled, emitted_at_cancel, emitted_later, history, events

        received, cancelled, at_cancel, later, history, events = asyncio.run(scenario())
        assert received >= 2
        assert cancelled is True
        assert later == at_cancel, "tokens were still produced after cancellation"
        assert history["messages"] == []
        assert history["awaiting_reply"] is False
        assert events[-1]["event"] == "done"
        assert events[-1]["data"]["text"] == "recovered"


def test_a_timeout_is_reported_and_the_session_stays_usable(tmp_path: Path) -> None:
    slow_turn = ["x"] * 30
    with live_server(
        tmp_path, turns=[slow_turn, ["fine"]], delay=0.03, default_deadline=5.0
    ) as (url, model):
        async def scenario():
            async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                session = await _create_session(client)
                sid = session["session_id"]
                async with client.stream(
                    "POST",
                    f"/sessions/{sid}/messages",
                    json={"message": "too slow", "deadline_seconds": 0.1},
                    headers=ALICE,
                ) as response:
                    first = await _read_events(response)
                history = (await client.get(f"/sessions/{sid}", headers=ALICE)).json()
                async with client.stream(
                    "POST",
                    f"/sessions/{sid}/messages",
                    json={"message": "now"},
                    headers=ALICE,
                ) as response:
                    second = await _read_events(response)
            return first, history, second

        first, history, second = asyncio.run(scenario())
        assert first[-1]["event"] == "error"
        assert first[-1]["data"]["code"] == "timeout"
        assert "deadline" in first[-1]["data"]["message"]
        assert model.cancelled is True
        assert history["messages"] == []
        assert second[-1]["event"] == "done"
        assert second[-1]["data"]["text"] == "fine"


def test_a_second_concurrent_turn_is_refused(tmp_path: Path) -> None:
    long_turn = [f"t{i} " for i in range(100)]
    with live_server(tmp_path, turns=[long_turn, ["later"]], delay=0.02) as (url, _):
        async def scenario():
            async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                session = await _create_session(client)
                sid = session["session_id"]
                async with client.stream(
                    "POST",
                    f"/sessions/{sid}/messages",
                    json={"message": "first"},
                    headers=ALICE,
                ) as response:
                    async for event in _events(response):
                        if event["event"] == "token":
                            break
                    second = await client.post(
                        f"/sessions/{sid}/messages",
                        json={"message": "second"},
                        headers=ALICE,
                    )
                return second

        second = asyncio.run(scenario())
        assert second.status_code == 409


def test_a_resumed_conversation_keeps_its_context(tmp_path: Path) -> None:
    long_turn = [f"x{i} " for i in range(100)]
    with live_server(
        tmp_path,
        turns=[["My ", "name is Alice."], long_turn, ["I ", "remember."]],
        delay=0.02,
    ) as (url, model):
        async def scenario():
            async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                session = await _create_session(client)
                sid = session["session_id"]
                async with client.stream(
                    "POST",
                    f"/sessions/{sid}/messages",
                    json={"message": "remember my name"},
                    headers=ALICE,
                ) as response:
                    first = await _read_events(response)
                async with client.stream(
                    "POST",
                    f"/sessions/{sid}/messages",
                    json={"message": "lost turn"},
                    headers=ALICE,
                ) as response:
                    count = 0
                    async for event in _events(response):
                        if event["event"] == "token":
                            count += 1
                        if count >= 2:
                            break
                await _wait_until(lambda: model.cancelled)
                async with client.stream(
                    "POST",
                    f"/sessions/{sid}/messages",
                    json={"message": "what is my name?"},
                    headers=ALICE,
                ) as response:
                    third = await _read_events(response)
                history = (await client.get(f"/sessions/{sid}", headers=ALICE)).json()
            return first, third, history

        first, third, history = asyncio.run(scenario())
        assert first[-1]["data"]["text"] == "My name is Alice."
        assert third[-1]["event"] == "done"
        context = model.contexts[-1]
        assert "remember my name" in context
        assert "My name is Alice." in context
        assert "lost turn" not in context
        assert [m["content"] for m in history["messages"]] == [
            "remember my name",
            "My name is Alice.",
            "what is my name?",
            "I remember.",
        ]


def test_invalid_messages_and_deadlines_are_rejected(tmp_path: Path) -> None:
    with live_server(tmp_path, turns=[]) as (url, _):
        async def scenario():
            async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                session = await _create_session(client)
                sid = session["session_id"]
                empty = await client.post(
                    f"/sessions/{sid}/messages", json={"message": "   "}, headers=ALICE
                )
                oversized = await client.post(
                    f"/sessions/{sid}/messages",
                    json={"message": "x" * 8001},
                    headers=ALICE,
                )
                bad_deadline = await client.post(
                    f"/sessions/{sid}/messages",
                    json={"message": "hi", "deadline_seconds": 0.001},
                    headers=ALICE,
                )
            return empty, oversized, bad_deadline

        empty, oversized, bad_deadline = asyncio.run(scenario())
        assert empty.status_code == 422
        assert oversized.status_code == 422
        assert bad_deadline.status_code == 422


def test_delete_removes_the_session(tmp_path: Path) -> None:
    with live_server(tmp_path, turns=[]) as (url, _):
        async def scenario():
            async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                session = await _create_session(client)
                sid = session["session_id"]
                deleted = await client.delete(f"/sessions/{sid}", headers=ALICE)
                after = await client.get(f"/sessions/{sid}", headers=ALICE)
                listed = await client.get("/sessions", headers=ALICE)
            return deleted, after, listed

        deleted, after, listed = asyncio.run(scenario())
        assert deleted.status_code == 204
        assert after.status_code == 404
        assert listed.json()["sessions"] == []


def test_health_does_not_require_a_token(tmp_path: Path) -> None:
    with live_server(tmp_path, turns=[]) as (url, _):
        async def scenario():
            async with httpx.AsyncClient(base_url=url) as client:
                return await client.get("/healthz")

        response = asyncio.run(scenario())
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
