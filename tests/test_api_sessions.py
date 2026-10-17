"""Session service: labels, opaque ids and the user boundary.

These tests do not start a server. They pin the rules the HTTP layer relies
on: a session id is scoped to one user, and another user's id resolves to
nothing rather than to their conversation.
"""

import asyncio
from pathlib import Path

import pytest

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.api.sessions import SessionNotFoundError, SessionService
from gem_agentic_ai.persistence import DurableMemory, ThreadAccessError
from tests.fakes import StreamingChatModel


@pytest.fixture
def memory(tmp_path: Path) -> DurableMemory:
    store = DurableMemory(tmp_path / "memory")
    yield store
    store.close()


@pytest.fixture
def sessions(memory: DurableMemory) -> SessionService:
    return SessionService(memory)


def test_a_labelled_session_is_created_with_an_opaque_id(
    sessions: SessionService,
) -> None:
    record = sessions.create("alice", "support-1")
    assert record.thread_key == "support-1"
    assert record.user_id == "alice"
    assert record.storage_id != "support-1"
    assert len(record.storage_id) == 32


def test_creating_the_same_label_twice_reuses_the_session(
    sessions: SessionService,
) -> None:
    first = sessions.create("alice", "support-1")
    second = sessions.create("alice", "support-1")
    assert first.storage_id == second.storage_id


def test_a_session_without_a_label_gets_a_generated_key(
    sessions: SessionService,
) -> None:
    first = sessions.create("alice")
    second = sessions.create("alice")
    assert first.thread_key.startswith("session-")
    assert first.storage_id != second.storage_id


@pytest.mark.parametrize("label", ["", "has space", "x" * 65, "semi;colon", "../escape"])
def test_invalid_labels_are_refused(sessions: SessionService, label: str) -> None:
    with pytest.raises(ValueError):
        sessions.create("alice", label)


def test_another_user_cannot_resolve_the_session(sessions: SessionService) -> None:
    record = sessions.create("alice", "private")
    with pytest.raises(SessionNotFoundError):
        sessions.resolve("bob", record.storage_id)


def test_an_unknown_session_id_resolves_to_nothing(sessions: SessionService) -> None:
    with pytest.raises(SessionNotFoundError):
        sessions.resolve("alice", "does-not-exist")


def test_the_config_for_another_users_session_is_refused(
    memory: DurableMemory, sessions: SessionService
) -> None:
    record = sessions.create("alice", "private")
    with pytest.raises(ThreadAccessError):
        memory.config_for_session("bob", record.storage_id)


def test_listing_returns_only_the_callers_sessions(
    sessions: SessionService,
) -> None:
    sessions.create("alice", "a1")
    sessions.create("alice", "a2")
    sessions.create("bob", "b1")
    assert {r.thread_key for r in sessions.list("alice")} == {"a1", "a2"}
    assert {r.thread_key for r in sessions.list("bob")} == {"b1"}


def test_delete_is_scoped_to_the_owner(sessions: SessionService) -> None:
    record = sessions.create("alice", "gone")
    assert sessions.delete("bob", record.storage_id) is False
    assert sessions.resolve("alice", record.storage_id).thread_key == "gone"
    assert sessions.delete("alice", record.storage_id) is True
    with pytest.raises(SessionNotFoundError):
        sessions.resolve("alice", record.storage_id)


def test_a_new_session_has_no_history(sessions: SessionService) -> None:
    record = sessions.create("alice")
    messages, awaiting = sessions.history("alice", record.storage_id)
    assert messages == []
    assert awaiting is False


def test_history_returns_the_completed_turn(
    memory: DurableMemory, sessions: SessionService
) -> None:
    record = sessions.create("alice")
    config = sessions.config("alice", record.storage_id)
    model = StreamingChatModel(turns=[["Hello", " there"]])
    agent = build_agent(model, [], checkpointer=memory.checkpointer)
    asyncio.run(
        agent.ainvoke(
            {"messages": [{"role": "user", "content": "hi"}]},
            config,
        )
    )

    messages, awaiting = sessions.history("alice", record.storage_id)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hi"),
        ("assistant", "Hello there"),
    ]
    assert awaiting is False


def test_history_reports_an_unanswered_turn(
    memory: DurableMemory, sessions: SessionService
) -> None:
    """A human message with no reply is surfaced, not hidden."""
    record = sessions.create("alice")
    config = sessions.config("alice", record.storage_id)
    model = StreamingChatModel(turns=[["unused"]])
    agent = build_agent(model, [], checkpointer=memory.checkpointer)
    agent.update_state(config, {"messages": [{"role": "user", "content": "orphan"}]})

    messages, awaiting = sessions.history("alice", record.storage_id)
    assert [(m.role, m.content) for m in messages] == [("user", "orphan")]
    assert awaiting is True
