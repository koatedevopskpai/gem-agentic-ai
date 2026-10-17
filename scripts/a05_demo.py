"""End-to-end client demo for the async streaming API (A05).

It starts the real application under uvicorn on a loopback port and drives it
over HTTP with httpx. Five scenarios run in order:

1. latency      — 20 turns, client-side first-token and total latency
2. disconnect   — the client stops reading mid-turn and the *model* stops
3. timeout      — a turn exceeds its deadline and is cancelled
4. resume       — the session survives the abandoned turn with clean history
5. ownership    — another user gets 404 for every session operation

The model is a scripted double, not a provider: this makes the run free and
deterministic, and it is the same double the offline tests use. Latency here
is therefore the API layer's latency over a scripted stream, not model or
network latency; the figures say so where they are reported.

Run from the repository root, with the package importable::

    python scripts/a05_demo.py --out artifacts/a05_demo.json

Exit status is 0 only when every scenario's checks pass.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import socket
import statistics
import sys
import threading
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr

from gem_agentic_ai.api.server import create_app
from gem_agentic_ai.persistence import DurableMemory

TOKENS = {"demo-alice": "alice", "demo-bob": "bob"}
ALICE = {"Authorization": "Bearer demo-alice"}
BOB = {"Authorization": "Bearer demo-bob"}


class DemoStreamingModel(BaseChatModel):
    """Replay token lists, asynchronously, recording whether it was stopped.

    The same shape as the test double: a real scripted ``BaseChatModel`` so
    LangGraph treats it like a provider stream.
    """

    turns: list[list[str]]
    delay: float = 0.0
    contexts: list[list[str]] = Field(default_factory=list)
    emitted: int = 0
    cancelled: bool = False
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "demo-scripted-streaming"

    def _next_turn(self, messages: list[BaseMessage]) -> list[str]:
        if self._cursor >= len(self.turns):
            raise AssertionError("the demo model ran out of scripted turns")
        self.contexts.append([str(message.content) for message in messages])
        turn = self.turns[self._cursor]
        self._cursor += 1
        return turn

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> ChatResult:
        turn = self._next_turn(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="".join(turn)))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ):
        turn = self._next_turn(messages)
        finished = False
        try:
            for token in turn:
                if self.delay:
                    await asyncio.sleep(self.delay)
                self.emitted += 1
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
            finished = True
        finally:
            if not finished:
                self.cancelled = True

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> DemoStreamingModel:
        return self


@contextlib.contextmanager
def serve(app: FastAPI):
    """Run uvicorn on an ephemeral loopback port in a background thread."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", access_log=False, lifespan="off")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
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


async def events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Parse the SSE frames of one response."""
    current: dict[str, Any] = {}
    async for line in response.aiter_lines():
        if line.startswith("event: "):
            current["event"] = line[len("event: ") :]
        elif line.startswith("data: "):
            current["data"] = json.loads(line[len("data: ") :])
        elif line == "" and current:
            yield current
            current = {}


async def read_events(response: httpx.Response) -> list[dict[str, Any]]:
    return [event async for event in events(response)]


async def create_session(client: httpx.AsyncClient) -> str:
    response = await client.post("/sessions", json={}, headers=ALICE)
    response.raise_for_status()
    return response.json()["session_id"]


async def abandon_turn(url: str, sid: str, message: str, *, tokens: int = 2) -> int:
    """Stop reading a turn early and close its connection.

    A dedicated client is deliberate: a client that abandons an SSE response
    must close the connection before it sends anything else. Reusing a
    connection whose response body was only partly read delivers the rest of
    that body as the next response — the resume scenario once read the
    abandoned turn's own tokens as its answer for exactly this reason.
    """
    received = 0
    async with httpx.AsyncClient(base_url=url, timeout=30) as abandoned:
        async with abandoned.stream(
            "POST",
            f"/sessions/{sid}/messages",
            json={"message": message},
            headers=ALICE,
        ) as response:
            async for event in events(response):
                if event["event"] == "token":
                    received += 1
                if received >= tokens:
                    break
    return received


async def open_turn(
    client: httpx.AsyncClient, sid: str, message: str, *, timeout: float = 5.0
) -> httpx.Response:
    """POST a turn, retrying while the previous turn finishes cleaning up.

    A session stays claimed until the cancelled worker has completed its
    rollback, so a client that resumes immediately after a disconnect can see
    ``409`` for a few milliseconds. Waiting here is what makes the resume
    scenario deterministic: the next turn cannot race the previous cleanup.
    """
    deadline = time.time() + timeout
    while True:
        response = await client.send(
            client.build_request(
                "POST",
                f"/sessions/{sid}/messages",
                json={"message": message},
                headers=ALICE,
            ),
            stream=True,
        )
        if response.status_code != 409:
            return response
        await response.aclose()
        if time.time() > deadline:
            raise AssertionError("the session stayed claimed after its previous turn")
        await asyncio.sleep(0.01)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


async def scenario_latency(client: httpx.AsyncClient, turns: int) -> dict[str, Any]:
    """Client-side latency for a completed turn, over a scripted stream."""
    sid = await create_session(client)
    first_token: list[float] = []
    totals: list[float] = []
    server_first: list[float] = []
    server_total: list[float] = []
    for i in range(turns):
        started = time.perf_counter()
        first_at: float | None = None
        async with client.stream(
            "POST",
            f"/sessions/{sid}/messages",
            json={"message": f"latency turn {i}"},
            headers=ALICE,
        ) as response:
            response.raise_for_status()
            done: dict[str, Any] | None = None
            async for event in events(response):
                if event["event"] == "token" and first_at is None:
                    first_at = time.perf_counter()
                elif event["event"] == "done":
                    done = event["data"]
        finished = time.perf_counter()
        assert first_at is not None and done is not None
        first_token.append((first_at - started) * 1000)
        totals.append((finished - started) * 1000)
        server_first.append(done["timing"]["first_token_ms"])
        server_total.append(done["timing"]["total_ms"])

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "n": len(values),
            "median_ms": round(statistics.median(values), 2),
            "p95_ms": round(_percentile(values, 0.95), 2),
            "min_ms": round(min(values), 2),
            "max_ms": round(max(values), 2),
        }

    client_first = summary(first_token)
    client_total = summary(totals)
    server_first = summary(server_first)
    server_total = summary(server_total)
    return {
        "turns": turns,
        "client_first_token": client_first,
        "client_total": client_total,
        "server_first_token": server_first,
        "server_total": server_total,
        "overhead_ms": {
            "first_token": round(
                client_first["median_ms"] - server_first["median_ms"], 2
            ),
            "total": round(client_total["median_ms"] - server_total["median_ms"], 2),
        },
        "note": (
            "Scripted model streaming one token every 50 ms. The medians are "
            "dominated by that cadence; 'overhead_ms' is the client-observed "
            "time beyond the server's own turn timing (request handling, SSE "
            "transport and parsing). This is not provider latency."
        ),
    }


async def scenario_disconnect(
    url: str, client: httpx.AsyncClient, model: DemoStreamingModel
) -> dict[str, Any]:
    """Break the connection mid-turn; the model must stop, not just the socket."""
    sid = await create_session(client)
    emitted_before = model.emitted
    received = await abandon_turn(url, sid, "abandon me", tokens=3)

    cancelled = False
    deadline = time.time() + 5
    while time.time() < deadline and not cancelled:
        if model.cancelled:
            cancelled = True
        else:
            await asyncio.sleep(0.01)
    emitted_at_cancel = model.emitted
    await asyncio.sleep(0.3)
    emitted_later = model.emitted

    history = (await client.get(f"/sessions/{sid}", headers=ALICE)).json()
    opened = await open_turn(client, sid, "try again")
    async with contextlib.aclosing(opened) as response:
        follow_up = await read_events(response)
    return {
        "tokens_received_before_disconnect": received,
        "model_cancelled": cancelled,
        "tokens_this_turn_at_cancel": emitted_at_cancel - emitted_before,
        "tokens_this_turn_after_300ms": emitted_later - emitted_before,
        "history_after_disconnect": history["messages"],
        "awaiting_reply": history["awaiting_reply"],
        "follow_up": follow_up[-1]["data"]["text"],
        "checks": {
            "socket_closed": received >= 3,
            "model_stopped": cancelled and emitted_later == emitted_at_cancel,
            "question_rolled_back": history["messages"] == [],
            "session_usable_afterwards": follow_up[-1]["event"] == "done",
        },
    }


async def scenario_timeout(client: httpx.AsyncClient, model: DemoStreamingModel) -> dict[str, Any]:
    sid = await create_session(client)
    async with client.stream(
        "POST",
        f"/sessions/{sid}/messages",
        json={"message": "too slow", "deadline_seconds": 0.15},
        headers=ALICE,
    ) as response:
        frames = await read_events(response)
    history = (await client.get(f"/sessions/{sid}", headers=ALICE)).json()
    last = frames[-1]
    return {
        "final_event": last["event"],
        "code": last["data"].get("code"),
        "message": last["data"].get("message"),
        "tokens_before_error": len([f for f in frames if f["event"] == "token"]),
        "model_cancelled": model.cancelled,
        "history": history["messages"],
        "checks": {
            "reported_timeout": last["event"] == "error" and last["data"].get("code") == "timeout",
            "model_cancelled": model.cancelled,
            "question_rolled_back": history["messages"] == [],
        },
    }


async def scenario_resume(
    url: str, client: httpx.AsyncClient, model: DemoStreamingModel
) -> dict[str, Any]:
    sid = await create_session(client)
    async with client.stream(
        "POST",
        f"/sessions/{sid}/messages",
        json={"message": "My name is Alice."},
        headers=ALICE,
    ) as response:
        first = await read_events(response)

    await abandon_turn(url, sid, "this turn will be abandoned")
    deadline = time.time() + 5
    while time.time() < deadline and not model.cancelled:
        await asyncio.sleep(0.01)

    opened = await open_turn(client, sid, "What is my name?")
    async with contextlib.aclosing(opened) as response:
        third = await read_events(response)
    history = (await client.get(f"/sessions/{sid}", headers=ALICE)).json()
    context = model.contexts[-1]
    return {
        "first_answer": first[-1]["data"]["text"],
        "resumed_answer": third[-1]["data"]["text"],
        "model_context": context,
        "history": [(m["role"], m["content"]) for m in history["messages"]],
        "checks": {
            "first_turn_completed": first[-1]["event"] == "done",
            "abandoned_turn_absent_from_context": "this turn will be abandoned" not in context,
            "first_turn_present_in_context": "My name is Alice." in context,
            "session_resumed": third[-1]["event"] == "done",
        },
    }


async def scenario_ownership(client: httpx.AsyncClient) -> dict[str, Any]:
    sid = await create_session(client)
    get_other = await client.get(f"/sessions/{sid}", headers=BOB)
    post_other = await client.post(
        f"/sessions/{sid}/messages", json={"message": "steal"}, headers=BOB
    )
    delete_other = await client.delete(f"/sessions/{sid}", headers=BOB)
    listed = (await client.get("/sessions", headers=BOB)).json()["sessions"]
    return {
        "get_status": get_other.status_code,
        "post_status": post_other.status_code,
        "delete_status": delete_other.status_code,
        "bob_session_count": len(listed),
        "checks": {
            "get_is_404": get_other.status_code == 404,
            "post_is_404": post_other.status_code == 404,
            "delete_is_404": delete_other.status_code == 404,
            "not_listed_for_other_user": listed == [],
        },
    }


async def scenario_validation(client: httpx.AsyncClient) -> dict[str, Any]:
    sid = await create_session(client)
    empty = await client.post(f"/sessions/{sid}/messages", json={"message": "   "}, headers=ALICE)
    deadline = await client.post(
        f"/sessions/{sid}/messages",
        json={"message": "hi", "deadline_seconds": 0.001},
        headers=ALICE,
    )
    return {
        "empty_message_status": empty.status_code,
        "bad_deadline_status": deadline.status_code,
        "checks": {
            "empty_is_422": empty.status_code == 422,
            "bad_deadline_is_422": deadline.status_code == 422,
        },
    }


async def run(directory: Path, latency_turns: int) -> dict[str, Any]:
    """Start the server, run every scenario, and return the report."""
    memory = DurableMemory(directory / "memory")
    long_turn = [f"t{i} " for i in range(200)]
    latency_turn = ["Hel", "lo", " from", " the", " stream", "."]
    model = DemoStreamingModel(
        turns=[latency_turn] * latency_turns
        + [long_turn, ["recovered"]]
        + [long_turn]
        + [["My ", "name is Alice."], long_turn, ["I ", "remember."]],
        # Above the default Windows timer resolution (about 15.6 ms): a
        # smaller sleep on the Proactor loop can return without waiting.
        delay=0.05,
    )
    app = create_app(llm=model, tokens=TOKENS, memory=memory, default_deadline_seconds=5.0)
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "modalities": "text only",
        "model": "scripted streaming double (no provider call)",
    }
    try:
        with serve(app) as url:
            report["base_url"] = url
            async with httpx.AsyncClient(base_url=url, timeout=30) as client:
                report["latency"] = await scenario_latency(client, latency_turns)
                report["disconnect"] = await scenario_disconnect(url, client, model)
                # ``cancelled`` latches on the model, so clear it between the
                # scenarios that read it; each one must prove its own turn
                # was stopped.
                model.cancelled = False
                report["timeout"] = await scenario_timeout(client, model)
                model.cancelled = False
                report["resume"] = await scenario_resume(url, client, model)
                report["ownership"] = await scenario_ownership(client)
                report["validation"] = await scenario_validation(client)
    finally:
        memory.close()

    checks: dict[str, bool] = {}
    for name, scenario in report.items():
        if isinstance(scenario, dict) and "checks" in scenario:
            for check, passed in scenario["checks"].items():
                checks[f"{name}.{check}"] = bool(passed)
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("a05_demo.json"))
    parser.add_argument("--latency-turns", type=int, default=20)
    parser.add_argument("--workdir", type=Path, default=Path("var/a05-demo"))
    args = parser.parse_args()

    report = asyncio.run(run(args.workdir, args.latency_turns))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    latency = report["latency"]
    print(f"generated: {args.out}")
    print(
        "latency (client): first token median "
        f"{latency['client_first_token']['median_ms']} ms, "
        f"p95 {latency['client_first_token']['p95_ms']} ms; "
        f"total median {latency['client_total']['median_ms']} ms, "
        f"p95 {latency['client_total']['p95_ms']} ms"
    )
    for name, scenario in report.items():
        if isinstance(scenario, dict) and "checks" in scenario:
            passed = all(scenario["checks"].values())
            print(f"{name}: {'PASS' if passed else 'FAIL'} ({scenario['checks']})")
    print(f"overall: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
