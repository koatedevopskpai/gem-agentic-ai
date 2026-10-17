"""Test doubles for the agent execution tests.

WHY a scripted chat model rather than ``MagicMock``?
  The agent calls ``.invoke()`` on a LangChain chat model and expects a real
  ``AIMessage``, including ``tool_calls``. A scripted ``BaseChatModel`` is a
  real Runnable, so the same object works with the manual ReAct loop and
  with LangGraph's ``create_agent``; a ``MagicMock`` only fits the former.

  It also records every message list it receives, which is how the tests
  prove that tool observations were fed back to the model.
"""

import asyncio
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr


def tool_call(name: str, args: dict | None = None, call_id: str = "call_1") -> dict:
    """Build a tool-call entry in the shape LangChain puts on ``AIMessage``."""
    return {"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """Replay a fixed list of ``AIMessage`` responses, recording the input.

    Attributes:
        responses: The messages to return, one per model turn, in order.
        seen: Every message list the model was invoked with. ``seen[n]`` is
            the context for turn ``n``.
    """

    responses: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: object,
    ) -> ChatResult:
        if self._cursor >= len(self.responses):
            raise AssertionError(
                f"ScriptedChatModel exhausted after {self._cursor} responses; "
                "the agent asked for another turn. Add a scripted response."
            )
        self.seen.append(list(messages))
        response = self.responses[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> "ScriptedChatModel":
        """Ignore the tool list: the scripted responses already contain calls."""
        return self


class StreamingChatModel(BaseChatModel):
    """Emit scripted token chunks asynchronously, one list per turn.

    A real provider streams ``AIMessageChunk`` objects; this double does the
    same, so the API's streaming, deadline and cancellation paths are
    exercised without a network call. It records the context of every turn
    and whether its stream was cancelled, which is how the tests distinguish
    "work stopped" from "the socket closed".

    Attributes:
        turns: Token lists, one per model turn, in order.
        delay: Seconds to await between tokens. Zero keeps tests fast; a
            small non-zero value leaves room for cancellation to arrive.
        contexts: The message contents the model was sent, per turn.
        emitted: Total tokens yielded across all turns.
        cancelled: True when a stream was cancelled before finishing.
    """

    turns: list[list[str]]
    delay: float = 0.0
    contexts: list[list[str]] = Field(default_factory=list)
    emitted: int = 0
    cancelled: bool = False
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "streaming-scripted"

    def _next_turn(self, messages: list[BaseMessage]) -> list[str]:
        if self._cursor >= len(self.turns):
            raise AssertionError(
                f"StreamingChatModel exhausted after {self._cursor} turns; "
                "add another scripted turn."
            )
        self.contexts.append([str(message.content) for message in messages])
        turn = self.turns[self._cursor]
        self._cursor += 1
        return turn

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: object,
    ) -> ChatResult:
        turn = self._next_turn(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="".join(turn)))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
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
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            # A stream closed early (``aclose``) raises GeneratorExit rather
            # than CancelledError; either way the turn did not finish, and
            # "stopped" is the property the tests need to observe.
            if not finished:
                self.cancelled = True

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> "StreamingChatModel":
        """Ignore the tool list: this double never emits tool calls."""
        return self
