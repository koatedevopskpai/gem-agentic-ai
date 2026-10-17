"""One streaming turn: token order, deadline, cancellation and rollback.

These tests drive ``stream_turn`` directly. The server tests
(``test_api_server.py``) prove the same paths over a real socket; these make
the cancellation and timeout semantics deterministic and fast.
"""

import asyncio
import contextlib
import json

import pytest
from langgraph.checkpoint.memory import MemorySaver

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.api.streaming import (
    DEFAULT_DEADLINE_SECONDS,
    ThreadedTurn,
    TurnEvent,
    encode_sse,
    resolve_deadline,
    stream_turn,
    validate_message,
)
from tests.fakes import StreamingChatModel


def _agent(model: StreamingChatModel, checkpointer=None):
    return build_agent(model, [], checkpointer=checkpointer or MemorySaver())


def _config(thread: str = "stream-1") -> dict:
    return {"configurable": {"thread_id": thread}}


async def _collect(agent, config, text: str, **kwargs) -> list[TurnEvent]:
    return [event async for event in stream_turn(agent, config, text, **kwargs)]


def test_tokens_are_forwarded_in_order_and_then_done() -> None:
    model = StreamingChatModel(turns=[["Hel", "lo", " world"]])

    async def scenario() -> list[TurnEvent]:
        return await _collect(_agent(model), _config(), "hi")

    events = asyncio.run(scenario())
    assert [event.kind for event in events] == ["token", "token", "token", "done"]
    assert [event.text for event in events if event.kind == "token"] == [
        "Hel",
        "lo",
        " world",
    ]
    done = events[-1]
    assert done.text == "Hello world"
    assert done.timing is not None
    assert done.timing.tokens == 3
    assert done.timing.first_token_ms is not None
    assert done.timing.total_ms >= done.timing.first_token_ms


def test_the_first_token_latency_reflects_model_delay() -> None:
    model = StreamingChatModel(turns=[["a", "b"]], delay=0.03)

    async def scenario():
        return await _collect(_agent(model), _config(), "hi")

    done = asyncio.run(scenario())[-1]
    assert done.timing is not None
    assert done.timing.first_token_ms >= 25
    assert done.timing.total_ms >= done.timing.first_token_ms


def test_cancelling_the_consumer_stops_the_model_and_rolls_back() -> None:
    """A closed socket is not enough: the run itself must be cancelled."""
    model = StreamingChatModel(turns=[["t0", "t1", "t2", "t3", "t4"]], delay=0.02)
    checkpointer = MemorySaver()
    agent = _agent(model, checkpointer)
    config = _config("cancel-1")
    seen: list[TurnEvent] = []

    async def scenario() -> None:
        async def consume() -> None:
            async for event in stream_turn(agent, config, "cancel me"):
                seen.append(event)

        task = asyncio.create_task(consume())
        while len([e for e in seen if e.kind == "token"]) < 2:
            await asyncio.sleep(0.005)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # The shielded rollback finishes on the loop after cancellation.
        await asyncio.sleep(0.05)
        emitted_at_cancel = model.emitted
        await asyncio.sleep(0.1)
        assert model.emitted == emitted_at_cancel

        state = await agent.aget_state(config)
        messages = state.values.get("messages", []) if state else []
        assert messages == []

    asyncio.run(scenario())
    assert model.cancelled is True
    assert len(seen) < 5


def test_a_threaded_turn_crosses_threads_and_cancels_the_model() -> None:
    """The production runner: events reach the request loop, and ``cancel``
    stops the model and rolls the unanswered question back.

    This is the path the HTTP endpoint uses. It is tested here, on a plain
    loop, so the semantics are deterministic; ``test_api_server`` exercises
    the same class behind a real ASGI server.
    """
    model = StreamingChatModel(turns=[["t0", "t1", "t2", "t3", "t4"]], delay=0.02)
    checkpointer = MemorySaver()
    agent = _agent(model, checkpointer)
    config = _config("threaded-cancel-1")
    seen: list[TurnEvent] = []

    async def scenario() -> None:
        turn = ThreadedTurn(agent, config, "cancel me").start(asyncio.get_running_loop())
        try:
            while len([e for e in seen if e.kind == "token"]) < 2:
                event = await asyncio.wait_for(turn.events.get(), timeout=5)
                assert event is not None
                seen.append(event)
        finally:
            turn.cancel()
        # Drain to the worker's sentinel: it is published after cancellation
        # handling, including the rollback, has finished.
        while True:
            event = await asyncio.wait_for(turn.events.get(), timeout=5)
            if event is None:
                break
            seen.append(event)

        state = await agent.aget_state(config)
        messages = state.values.get("messages", []) if state else []
        assert messages == []

    asyncio.run(scenario())
    tokens = [event.text for event in seen if event.kind == "token"]
    assert tokens[:2] == ["t0", "t1"]
    assert len(tokens) < 5
    assert model.cancelled is True
    # Cancellation reached the model: it did not run the remaining tokens.
    assert model.emitted < 5


def test_a_timeout_is_reported_and_stops_the_model() -> None:
    model = StreamingChatModel(turns=[["x"] * 20], delay=0.03)
    checkpointer = MemorySaver()
    agent = _agent(model, checkpointer)
    config = _config("timeout-1")

    async def scenario():
        events = await _collect(agent, config, "too slow", deadline_seconds=0.1)
        state = await agent.aget_state(config)
        return events, state

    events, state = asyncio.run(scenario())
    assert events[-1].kind == "error"
    assert events[-1].code == "timeout"
    assert "deadline" in events[-1].message
    assert model.cancelled is True
    assert model.emitted < 20
    assert state.values.get("messages", []) == []


def test_a_model_failure_is_reported_and_rolled_back() -> None:
    model = StreamingChatModel(turns=[])
    checkpointer = MemorySaver()
    agent = _agent(model, checkpointer)
    config = _config("error-1")

    async def scenario():
        events = await _collect(agent, config, "boom")
        state = await agent.aget_state(config)
        return events, state

    events, state = asyncio.run(scenario())
    assert events[-1].kind == "error"
    assert events[-1].code == "agent_error"
    assert "exhausted" in events[-1].message
    assert state.values.get("messages", []) == []


def test_the_next_turn_after_a_cancellation_sees_clean_history() -> None:
    model = StreamingChatModel(turns=[["partial ", "answer", " tail"], ["fine"]], delay=0.02)
    agent = _agent(model)
    config = _config("resume-1")

    async def scenario() -> list[str]:
        async def consume() -> None:
            async for _ in stream_turn(agent, config, "lost question"):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.06)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)

        events = await _collect(agent, config, "new question")
        assert events[-1].kind == "done"
        return model.contexts[-1]

    context = asyncio.run(scenario())
    assert context[-1] == "new question"
    assert "lost question" not in context


@pytest.mark.parametrize(
    "text",
    ["", "   ", "x" * 8001],
)
def test_invalid_messages_are_refused(text: str) -> None:
    with pytest.raises(ValueError):
        validate_message(text)


def test_a_valid_message_is_trimmed() -> None:
    assert validate_message("  hello  ") == "hello"


def test_deadlines_are_validated() -> None:
    assert resolve_deadline(None) == DEFAULT_DEADLINE_SECONDS
    assert resolve_deadline(2) == 2.0
    for bad in (0, -1, 0.01, 301, True):
        with pytest.raises(ValueError):
            resolve_deadline(bad)


def test_sse_frames_are_named_events_with_json_data() -> None:
    token = encode_sse(TurnEvent("token", text="hi"))
    assert token.startswith("event: token\ndata: ")
    assert token.endswith("\n\n")
    assert json.loads(token.split("data: ", 1)[1]) == {"text": "hi"}

    error = encode_sse(TurnEvent("error", code="timeout", message="slow"))
    assert error.startswith("event: error\n")
    payload = json.loads(error.split("data: ", 1)[1])
    assert payload["code"] == "timeout"
    assert payload["message"] == "slow"
