"""Durable, correctly scoped conversation memory.

WHY
  ``MemorySaver`` forgets a conversation when the process exits. This module
  stores LangGraph checkpoints in a local SQLite file and puts a user
  boundary around every thread, so a caller cannot resume another user's
  conversation by reusing or guessing a thread id.

WHAT IS DURABLE
  * checkpoint state (messages, pending writes) for each thread;
  * a thread registry with the owning user, creation time and last use;
  * completed tool actions, through ``SqliteIdempotencyStore`` (A03), so a
    replay after a restart does not repeat a side effect.

WHAT IS NOT DURABLE
  * approval decisions. They describe a human decision, not conversation
    state, and must be made again after a restart.
  * a run in progress. A checkpoint is written at step boundaries; a process
    killed mid-tool may leave the model's request without a matching result.
    LangGraph resumes from the last completed checkpoint, so the step is
    replayed. The idempotency store is what prevents a repeated side effect.
  * concurrent writers on different machines. This is a local SQLite file,
    not a distributed lock. Two processes on one host are serialised by
    SQLite; two hosts are not.

OWNERSHIP
  ``DurableMemory.config(user_id, thread_key)`` mints a config containing an
  opaque storage id and the user id. The scoped checkpointer refuses any
  config that is missing either value or whose pair is not in the registry.
  Storage ids are random, so knowing another user's thread key does not
  reveal or grant access to their id.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.sqlite import SqliteSaver

from gem_agentic_ai.bounds import ToolLimits
from gem_agentic_ai.guard import ApprovalStore, ToolGuard
from gem_agentic_ai.idempotency import SqliteIdempotencyStore

DEFAULT_RETENTION_DAYS = 30


class ThreadAccessError(PermissionError):
    """A config named a thread that the requesting user does not own."""


@dataclass(frozen=True)
class ThreadRecord:
    """One registered conversation thread."""

    user_id: str
    thread_key: str
    storage_id: str
    created_at: float
    last_used_at: float


def _require_scope(user_id: str, thread_key: str) -> tuple[str, str]:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if not isinstance(thread_key, str) or not thread_key.strip():
        raise ValueError("thread_key must be a non-empty string")
    return user_id.strip(), thread_key.strip()


class _ThreadRegistry:
    """Ownership and retention metadata, on its own SQLite connection."""

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS thread_registry ("
                "  storage_id TEXT PRIMARY KEY,"
                "  user_id TEXT NOT NULL,"
                "  thread_key TEXT NOT NULL,"
                "  created_at REAL NOT NULL,"
                "  last_used_at REAL NOT NULL,"
                "  UNIQUE (user_id, thread_key)"
                ")"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_registry_user_used "
                "ON thread_registry(user_id, last_used_at)"
            )
            self._conn.commit()

    def get_or_create(self, user_id: str, thread_key: str, now: float) -> ThreadRecord:
        with self._lock:
            # INSERT OR IGNORE plus the UNIQUE constraint makes two processes
            # racing on the same (user, thread) converge on one storage id.
            self._conn.execute(
                "INSERT OR IGNORE INTO thread_registry"
                "(storage_id, user_id, thread_key, created_at, last_used_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, user_id, thread_key, now, now),
            )
            self._conn.execute(
                "UPDATE thread_registry SET last_used_at = ? "
                "WHERE user_id = ? AND thread_key = ?",
                (now, user_id, thread_key),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT user_id, thread_key, storage_id, created_at, last_used_at "
                "FROM thread_registry WHERE user_id = ? AND thread_key = ?",
                (user_id, thread_key),
            ).fetchone()
        assert row is not None
        return ThreadRecord(*row)

    def find(self, user_id: str, thread_key: str) -> ThreadRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, thread_key, storage_id, created_at, last_used_at "
                "FROM thread_registry WHERE user_id = ? AND thread_key = ?",
                (user_id, thread_key),
            ).fetchone()
        return None if row is None else ThreadRecord(*row)

    def owns(self, user_id: str, storage_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM thread_registry WHERE storage_id = ? AND user_id = ?",
                (storage_id, user_id),
            ).fetchone()
        return row is not None

    def touch(self, storage_id: str, now: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE thread_registry SET last_used_at = ? WHERE storage_id = ?",
                (now, storage_id),
            )
            self._conn.commit()

    def list_for_user(self, user_id: str) -> tuple[ThreadRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, thread_key, storage_id, created_at, last_used_at "
                "FROM thread_registry WHERE user_id = ? ORDER BY last_used_at DESC",
                (user_id,),
            ).fetchall()
        return tuple(ThreadRecord(*row) for row in rows)

    def expired(self, cutoff: float) -> tuple[ThreadRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, thread_key, storage_id, created_at, last_used_at "
                "FROM thread_registry WHERE last_used_at < ? ORDER BY last_used_at",
                (cutoff,),
            ).fetchall()
        return tuple(ThreadRecord(*row) for row in rows)

    def delete(self, user_id: str, storage_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM thread_registry WHERE storage_id = ? AND user_id = ?",
                (storage_id, user_id),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class _ScopedCheckpointer(BaseCheckpointSaver):
    """A checkpointer that requires a registered ``(user_id, thread_id)`` pair.

    LangGraph passes the caller's config through to every checkpoint
    operation, so the user boundary can be enforced on read and write rather
    than being a convention at the call site.
    """

    def __init__(
        self,
        saver: BaseCheckpointSaver,
        registry: _ThreadRegistry,
        clock: Callable[[], float],
    ) -> None:
        super().__init__(serde=saver.serde)
        self._saver = saver
        self._registry = registry
        self._clock = clock

    @property
    def config_specs(self) -> list:
        return self._saver.config_specs

    def _authorise(self, config: RunnableConfig | None, *, touch: bool = False) -> str:
        configurable = (config or {}).get("configurable") or {}
        user_id = configurable.get("user_id")
        storage_id = configurable.get("thread_id")
        if not isinstance(user_id, str) or not user_id:
            raise ThreadAccessError(
                "durable memory requires config['configurable']['user_id']; "
                "build configs with DurableMemory.config()"
            )
        if not isinstance(storage_id, str) or not storage_id:
            raise ThreadAccessError(
                "durable memory requires config['configurable']['thread_id']; "
                "build configs with DurableMemory.config()"
            )
        if not self._registry.owns(user_id, storage_id):
            raise ThreadAccessError(
                f"thread '{storage_id}' is not owned by user '{user_id}'"
            )
        if touch:
            self._registry.touch(storage_id, self._clock())
        return storage_id

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self._authorise(config)
        return self._saver.get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        self._authorise(config)
        yield from self._saver.list(config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        self._authorise(config, touch=True)
        return self._saver.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._authorise(config, touch=True)
        self._saver.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        raise ThreadAccessError(
            "delete threads through DurableMemory.delete_thread(user_id, thread_key)"
        )


class DurableMemory:
    """Local durable checkpoint storage with user-scoped threads.

    Args:
        directory: Directory for ``checkpoints.sqlite`` and
            ``idempotency.sqlite``. Created if it does not exist.
        retention_days: Idle threads older than this are removed by
            ``sweep_expired``. Sweeping is explicit; opening the store never
            deletes anything.
        clock: Injectable clock for retention tests.

    Example:
        >>> memory = DurableMemory("var/agent-memory")
        >>> config = memory.config("alice", "support-1")
        >>> agent = build_agent(llm, tools, checkpointer=memory.checkpointer)
        >>> agent.invoke({"messages": [("user", "hello")]}, config)
        >>> memory.list_threads("alice")
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(retention_days, int) or retention_days <= 0:
            raise ValueError("retention_days must be a positive integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._clock = clock
        self._closed = False

        checkpoint_path = self.directory / "checkpoints.sqlite"
        self._conn = sqlite3.connect(
            checkpoint_path, check_same_thread=False, timeout=5.0
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._saver = SqliteSaver(self._conn)
        self._saver.setup()
        self._registry = _ThreadRegistry(self.directory / "thread-registry.sqlite")
        self._checkpointer = _ScopedCheckpointer(self._saver, self._registry, self._clock)

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        """The user-scoped checkpointer to pass to ``build_agent``."""
        return self._checkpointer

    def config(self, user_id: str, thread_key: str) -> dict[str, Any]:
        """Return a LangGraph config for one user's thread.

        The same ``(user_id, thread_key)`` pair resolves to the same opaque
        storage id across restarts.
        """
        user_id, thread_key = _require_scope(user_id, thread_key)
        record = self._registry.get_or_create(user_id, thread_key, self._clock())
        return {
            "configurable": {
                "thread_id": record.storage_id,
                "user_id": user_id,
            }
        }

    def list_threads(self, user_id: str) -> tuple[ThreadRecord, ...]:
        """Threads owned by ``user_id``, most recently used first."""
        user_id, _ = _require_scope(user_id, "_")
        return self._registry.list_for_user(user_id)

    def delete_thread(self, user_id: str, thread_key: str) -> bool:
        """Delete one thread's checkpoints and registry row.

        Returns ``False`` when the thread does not exist for this user. The
        user id is part of the lookup, so one user cannot delete another
        user's thread by reusing its key.
        """
        user_id, thread_key = _require_scope(user_id, thread_key)
        record = self._registry.find(user_id, thread_key)
        if record is None:
            return False
        self._saver.delete_thread(record.storage_id)
        self._registry.delete(user_id, record.storage_id)
        return True

    def sweep_expired(self, *, now: float | None = None) -> tuple[ThreadRecord, ...]:
        """Delete threads idle longer than ``retention_days``.

        Returns the deleted records so the caller can report or audit the
        sweep. This is the only operation that deletes data automatically.
        """
        current = self._clock() if now is None else now
        cutoff = current - (self.retention_days * 24 * 60 * 60)
        records = self._registry.expired(cutoff)
        for record in records:
            self._saver.delete_thread(record.storage_id)
            self._registry.delete(record.user_id, record.storage_id)
        return records

    def guard(
        self,
        limits: ToolLimits | None = None,
        *,
        approvals: ApprovalStore | None = None,
    ) -> ToolGuard:
        """Build a guard whose completed actions survive a restart.

        Approval decisions do not persist: they record a human decision and
        must be made again in a new process.
        """
        return ToolGuard(
            limits,
            approvals=approvals,
            idempotency=SqliteIdempotencyStore(self.directory / "idempotency.sqlite"),
        )

    def close(self) -> None:
        """Close both SQLite connections. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._registry.close()
        self._conn.close()

    def __enter__(self) -> DurableMemory:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
