"""
AGENT STATE DEFINITION
======================
The shared state schema passed through the LangGraph state machine.

WHY a TypedDict?
  TypedDict provides key-value access (like a plain dict) with
  static type checking. LangGraph passes state between its nodes
  as dict-like objects, so TypedDict is the natural fit. Pydantic
  BaseModel would also work but adds serialisation overhead that
  this demo does not need.

WHY Annotated[list, add_messages]?
  LangGraph uses Annotated types with reducer functions to define
  how state keys are updated. The add_messages reducer APPENDS new
  messages to the list rather than overwriting the entire list.
  This is essential for conversation — each turn adds to the history
  without losing previous turns.

  Without this reducer, each graph step would replace the message
  list, so the agent would lose context of what it said two turns ago.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The state flowing through the LangGraph agent graph.

    Every node in the graph can read and write to this state.
    The keys defined here are the ONLY keys the graph will manage —
    LangGraph ignores extra keys (though it won't error).

    Attributes:
        messages: Ordered list of conversation turns. Each turn
                  adds HumanMessage, AIMessage, and ToolMessage
                  objects. The add_messages reducer ensures
                  accumulation rather than replacement.
    """
    messages: Annotated[list, add_messages]
