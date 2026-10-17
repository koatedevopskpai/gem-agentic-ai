"""Tests for durable, user-scoped conversation memory.

Each test asserts on something the caller can observe across a process
boundary: the messages a reopened agent sees, the registry's ownership
records, and whether an action's ledger grew. No network, no model API.
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.bounds import ToolLimits
from gem_agentic_ai.persistence import DurableMemory, ThreadAccessError
from gem_agentic_ai.tools.actions import (
    create_support_ticket,
    reset_ticket_ledger,
    ticket_ledger,
)
from gem_agentic_ai.tools.system import calculate_multiplication
from tests.fakes import ScriptedChatModel


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def clean_ledger():
    reset_ticket_ledger()
    yield
    reset_ticket_ledger()


def _agent(memory: DurableMemory, responses: list[AIMessage]):
    model = ScriptedChatModel(responses=responses)
    agent = build_agent(
        model,
        [calculate_multiplication],
        checkpointer=memory.checkpointer,
    )
    return agent, model


def _ask(agent, config: dict, text: str) -> dict:
    return agent.invoke({"messages": [("user", text)]}, config)


def _contents(messages: list) -> list[str]:
    return [message.content for message in messages]


class TestRestartAndResume:
    def test_conversation_survives_a_process_restart(self, tmp_path: Path) -> None:
        directory = tmp_path / "store"

        with DurableMemory(directory) as memory:
            config = memory.config("alice", "support-1")
            agent, _ = _agent(memory, [AIMessage(content="Noted, Ada.")])
            _ask(agent, config, "My name is Ada.")
            storage_id = config["configurable"]["thread_id"]

        with DurableMemory(directory) as memory:
            reopened = memory.config("alice", "support-1")
            agent, model = _agent(memory, [AIMessage(content="Your name is Ada.")])
            result = _ask(agent, reopened, "What is my name?")

        assert reopened["configurable"]["thread_id"] == storage_id
        assert _contents(result["messages"]) == [
            "My name is Ada.",
            "Noted, Ada.",
            "What is my name?",
            "Your name is Ada.",
        ]
        # The second process sent the earlier turns to the model, which is
        # what "state survived" means from the model's point of view.
        assert _contents(model.seen[0]) == [
            "My name is Ada.",
            "Noted, Ada.",
            "What is my name?",
        ]

    def test_the_same_pair_resolves_to_the_same_storage_id(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            first = memory.config("alice", "support-1")
            second = memory.config("alice", "support-1")
            other = memory.config("bob", "support-1")

        assert first == second
        assert first["configurable"]["thread_id"] != other["configurable"]["thread_id"]


class TestOwnership:
    def _seed(self, memory: DurableMemory) -> dict:
        config = memory.config("alice", "support-1")
        agent, _ = _agent(memory, [AIMessage(content="Hello Alice.")])
        _ask(agent, config, "Hello.")
        return config

    def test_another_user_cannot_read_the_thread(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            config = self._seed(memory)
            tampered = {
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "user_id": "bob",
                }
            }

            with pytest.raises(ThreadAccessError):
                memory.checkpointer.get_tuple(tampered)

    def test_another_user_cannot_extend_the_thread(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            config = self._seed(memory)
            tampered = {
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "user_id": "bob",
                }
            }
            agent, model = _agent(memory, [AIMessage(content="unreachable")])

            with pytest.raises(ThreadAccessError):
                _ask(agent, tampered, "Read Alice's conversation.")

            assert model.seen == []

    def test_a_config_without_a_user_is_refused(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            config = self._seed(memory)
            unscoped = {
                "configurable": {"thread_id": config["configurable"]["thread_id"]}
            }

            with pytest.raises(ThreadAccessError, match="user_id"):
                memory.checkpointer.get_tuple(unscoped)

    def test_a_config_without_a_thread_is_refused(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            with pytest.raises(ThreadAccessError, match="thread_id"):
                memory.checkpointer.get_tuple(
                    {"configurable": {"user_id": "alice"}}
                )

    def test_an_unregistered_thread_id_is_refused(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            with pytest.raises(ThreadAccessError, match="not owned"):
                memory.checkpointer.get_tuple(
                    {"configurable": {"thread_id": "guessed-id", "user_id": "alice"}}
                )

    def test_listing_checkpoints_is_scoped(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            config = self._seed(memory)
            assert list(memory.checkpointer.list(config))
            tampered = {
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "user_id": "bob",
                }
            }
            with pytest.raises(ThreadAccessError):
                next(iter(memory.checkpointer.list(tampered)))

    def test_thread_list_shows_only_the_owners_threads(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            memory.config("alice", "support-1")
            memory.config("alice", "support-2")
            memory.config("bob", "support-1")

            records = memory.list_threads("alice")

        assert {record.thread_key for record in records} == {"support-1", "support-2"}
        assert {record.user_id for record in records} == {"alice"}


class TestThreadIsolation:
    def test_threads_do_not_share_state(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            first = memory.config("alice", "support-1")
            agent, _ = _agent(memory, [AIMessage(content="Ticket A noted.")])
            _ask(agent, first, "Conversation A")

            second = memory.config("alice", "support-2")
            agent, model = _agent(memory, [AIMessage(content="Ticket B noted.")])
            result = _ask(agent, second, "Conversation B")

        assert _contents(model.seen[0]) == ["Conversation B"]
        assert _contents(result["messages"]) == [
            "Conversation B",
            "Ticket B noted.",
        ]


class TestDeletion:
    def test_deleting_a_thread_removes_its_state(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            config = memory.config("alice", "support-1")
            agent, _ = _agent(memory, [AIMessage(content="Hello Alice.")])
            _ask(agent, config, "Hello.")
            storage_id = config["configurable"]["thread_id"]

            # Another user's key cannot delete it.
            assert memory.delete_thread("bob", "support-1") is False

            assert memory.delete_thread("alice", "support-1") is True
            assert memory.delete_thread("alice", "support-1") is False

            with pytest.raises(ThreadAccessError):
                memory.checkpointer.get_tuple(config)

            fresh = memory.config("alice", "support-1")
            agent, model = _agent(memory, [AIMessage(content="Starting over.")])
            result = _ask(agent, fresh, "A new conversation.")

        assert fresh["configurable"]["thread_id"] != storage_id
        assert _contents(model.seen[0]) == ["A new conversation."]
        assert _contents(result["messages"]) == [
            "A new conversation.",
            "Starting over.",
        ]

    def test_the_checkpointer_refuses_unscoped_deletion(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            config = memory.config("alice", "support-1")
            storage_id = config["configurable"]["thread_id"]

            with pytest.raises(ThreadAccessError, match="delete_thread"):
                memory.checkpointer.delete_thread(storage_id)

            assert memory.list_threads("alice")  # still registered


class TestRetention:
    def test_sweep_removes_only_idle_threads(self, tmp_path: Path) -> None:
        clock = FakeClock()
        with DurableMemory(tmp_path, retention_days=7, clock=clock) as memory:
            memory.config("alice", "active")
            memory.config("alice", "idle")
            memory.config("bob", "idle")

            clock.advance(6 * 24 * 60 * 60)
            memory.config("alice", "active")
            clock.advance(2 * 24 * 60 * 60)

            deleted = memory.sweep_expired()

            assert {(record.user_id, record.thread_key) for record in deleted} == {
                ("alice", "idle"),
                ("bob", "idle"),
            }
            assert [record.thread_key for record in memory.list_threads("alice")] == [
                "active"
            ]

    def test_opening_the_store_never_deletes_data(self, tmp_path: Path) -> None:
        clock = FakeClock()
        with DurableMemory(tmp_path, retention_days=1, clock=clock) as memory:
            memory.config("alice", "support-1")

        clock.advance(365 * 24 * 60 * 60)
        with DurableMemory(tmp_path, retention_days=1, clock=clock) as memory:
            assert [record.thread_key for record in memory.list_threads("alice")] == [
                "support-1"
            ]

    def test_retention_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="retention_days"):
            DurableMemory(tmp_path, retention_days=0)


class TestReplayAfterRestart:
    def test_a_completed_action_is_not_repeated(self, tmp_path: Path) -> None:
        args = {"summary": "printer offline", "idempotency_key": "ticket-1"}
        limits = ToolLimits(consequential_tools=frozenset({"create_support_ticket"}))

        with DurableMemory(tmp_path) as memory:
            guard = memory.guard(limits)
            guard.approvals.approve("create_support_ticket", "ticket-1")
            first = guard.execute(create_support_ticket, args)

        assert first.status == "ok"
        assert first.cached is False
        assert len(ticket_ledger()) == 1

        with DurableMemory(tmp_path) as memory:
            guard = memory.guard(limits)
            replay = guard.execute(create_support_ticket, args, call_id="new-call")

        assert replay.status == "ok"
        assert replay.cached is True
        assert len(ticket_ledger()) == 1

    def test_approval_is_not_durable(self, tmp_path: Path) -> None:
        args = {"summary": "printer offline", "idempotency_key": "ticket-2"}
        limits = ToolLimits(consequential_tools=frozenset({"create_support_ticket"}))

        with DurableMemory(tmp_path) as memory:
            guard = memory.guard(limits)
            guard.approvals.approve("create_support_ticket", "ticket-2")
            assert guard.execute(create_support_ticket, args).status == "ok"

        with DurableMemory(tmp_path) as memory:
            guard = memory.guard(limits)
            # A new key is presented as already approved in the old process.
            other = {"summary": "second request", "idempotency_key": "ticket-3"}
            outcome = guard.execute(create_support_ticket, other)

        assert outcome.status == "error"
        assert outcome.error_type == "approval_required"
        assert len(ticket_ledger()) == 1


class TestScopeValidation:
    def test_blank_user_or_thread_is_rejected(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            with pytest.raises(ValueError, match="user_id"):
                memory.config("  ", "support-1")
            with pytest.raises(ValueError, match="thread_key"):
                memory.config("alice", "")

    def test_config_specs_are_delegated(self, tmp_path: Path) -> None:
        with DurableMemory(tmp_path) as memory:
            assert isinstance(memory.checkpointer.config_specs, list)

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        memory = DurableMemory(tmp_path)
        memory.close()
        memory.close()
