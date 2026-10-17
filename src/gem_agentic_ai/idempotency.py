"""Idempotency records for side-effecting tool calls.

WHY this exists
  A retried or replayed agent run must not perform a side effect twice. The
  guard consults a store before executing a call and writes the completed
  result afterwards, keyed by ``(tool name, action key)``.

WHY two implementations
  ``InMemoryIdempotencyStore`` is enough for one process and keeps tests
  simple. ``SqliteIdempotencyStore`` survives a restart, which is what makes
  the "replay produces one logical result" claim testable across process
  boundaries. Neither is a distributed lock: two processes writing the same
  key at the same instant rely on SQLite's primary key plus
  ``INSERT OR IGNORE``, so the first committed result wins.

WHAT IS STORED
  Only the completed textual result. No arguments, no credentials and no
  provider payloads, so the ledger does not become a second copy of customer
  data.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol


class IdempotencyStore(Protocol):
    """Minimal interface the guard needs."""

    def get(self, tool: str, key: str) -> str | None: ...

    def put(self, tool: str, key: str, content: str) -> None: ...


class InMemoryIdempotencyStore:
    """Process-local record. Fast, and lost when the process exits."""

    def __init__(self) -> None:
        self._results: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def get(self, tool: str, key: str) -> str | None:
        with self._lock:
            return self._results.get((tool, key))

    def put(self, tool: str, key: str, content: str) -> None:
        with self._lock:
            self._results.setdefault((tool, key), content)


class SqliteIdempotencyStore:
    """Durable record in a local SQLite database.

    A fresh connection is opened per operation. That avoids passing a
    connection between threads and keeps the class safe to use from the
    guard's worker thread as well as the caller's thread.

    Args:
        path: Database file. The parent directory must exist; the guard does
            not create directories on the caller's behalf.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS idempotency ("
                "  tool TEXT NOT NULL,"
                "  key TEXT NOT NULL,"
                "  content TEXT NOT NULL,"
                "  created_at REAL NOT NULL,"
                "  PRIMARY KEY (tool, key)"
                ")"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0)

    def get(self, tool: str, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM idempotency WHERE tool = ? AND key = ?",
                (tool, key),
            ).fetchone()
        return None if row is None else str(row[0])

    def put(self, tool: str, key: str, content: str) -> None:
        # INSERT OR IGNORE keeps first-writer-wins: a replay must not replace
        # the result of the call that actually happened.
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO idempotency (tool, key, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (tool, key, content, time.time()),
            )
