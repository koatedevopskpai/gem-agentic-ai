"""Test doubles for the agent execution tests.

WHY a scripted chat model rather than ``MagicMock``?
  The agent calls ``.invoke()`` on a LangChain chat model and expects a real
  ``AIMessage``, including ``tool_calls``. A scripted ``BaseChatModel`` is a
  real Runnable, so the same object works with the manual ReAct loop and
  with LangGraph's ``create_agent``; a ``MagicMock`` only fits the former.

  It also records every message list it receives, which is how the tests
  prove that tool observations were fed back to the model.
"""

from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
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
