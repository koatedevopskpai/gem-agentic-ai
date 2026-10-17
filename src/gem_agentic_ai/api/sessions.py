"""Session service: the API's view of A02 durable memory.

A *session* is a durable LangGraph thread. The client is given the thread's
opaque storage id (the session id) and never sees the thread key, so the two
identifiers have different jobs:

* the **thread key** is the user's label for a conversation ("support-1");
* the **session id** is an unguessable storage handle, scoped to one user.

All ownership decisions go through ``DurableMemory``; this module does not
keep a second mapping that could drift out of step with the checkpointer.
A session id that belongs to another user resolves to nothing, which the API
renders as 404.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from gem_agentic_ai.persistence import DurableMemory, ThreadRecord

_LABEL = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class SessionNotFoundError(LookupError):
    """The session does not exist for this user (unknown or another user's)."""


@dataclass(frozen=True)
class MessageView:
    """One stored chat message, reduced to what a client needs to rehydrate."""

    role: str
    content: str
    tool_calls: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = list(self.tool_calls)
        return data


def _content_text(content: Any) -> str:
    """Render message content as text.

    The API is text-only (stated in the docs). A non-string content block is
    serialised rather than dropped so that a future multimodal message is
    visible as data instead of silently disappearing.
    """
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str, ensure_ascii=False)


def _view(message: BaseMessage) -> MessageView | None:
    if isinstance(message, HumanMessage):
        return MessageView("user", _content_text(message.content))
    if isinstance(message, AIMessage):
        calls = tuple(
            str(call.get("name", "")) for call in (message.tool_calls or [])
        )
        return MessageView("assistant", _content_text(message.content), calls)
    if isinstance(message, ToolMessage):
        return MessageView("tool", _content_text(message.content))
    return None


class SessionService:
    """Create, resolve, list and delete one user's sessions."""

    def __init__(self, memory: DurableMemory) -> None:
        self._memory = memory

    def create(self, user_id: str, label: str | None = None) -> ThreadRecord:
        """Create (or reuse) a session for ``user_id``.

        Reusing a label is deliberate: a client that retries "create" after a
        dropped response gets the same session instead of a second empty one.
        """
        if label is not None and not _LABEL.fullmatch(label):
            raise ValueError(
                "label must be 1-64 characters of letters, digits, '.', '_' or '-'"
            )
        thread_key = label or f"session-{uuid.uuid4().hex}"
        return self._memory.create_thread(user_id, thread_key)

    def list(self, user_id: str) -> tuple[ThreadRecord, ...]:
        return self._memory.list_threads(user_id)

    def resolve(self, user_id: str, session_id: str) -> ThreadRecord:
        record = self._memory.find_session(user_id, session_id)
        if record is None:
            raise SessionNotFoundError(session_id)
        return record

    def config(self, user_id: str, session_id: str) -> dict[str, Any]:
        self.resolve(user_id, session_id)
        return self._memory.config_for_session(user_id, session_id)

    def history(
        self, user_id: str, session_id: str
    ) -> tuple[list[MessageView], bool]:
        """Return the stored messages and whether a user turn awaits a reply.

        ``awaiting_reply`` is true when the last stored message is the user's
        and no assistant message followed it. The streaming layer rolls back
        cancelled and timed-out turns, so this normally means the process died
        between checkpointing the question and answering it; a reconnecting
        client should resend.
        """
        config = self.config(user_id, session_id)
        checkpoint = self._memory.checkpointer.get_tuple(config)
        if checkpoint is None:
            return [], False
        messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
        views = [view for view in (_view(m) for m in messages) if view is not None]
        awaiting = bool(views) and views[-1].role == "user"
        return views, awaiting

    def delete(self, user_id: str, session_id: str) -> bool:
        record = self._memory.find_session(user_id, session_id)
        if record is None:
            return False
        return self._memory.delete_thread(user_id, record.thread_key)
