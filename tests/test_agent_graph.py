"""
TESTS: LANGGRAPH AGENT BUILDER
==============================
Tests the build_agent function that constructs the LangGraph state machine.

We verify:
  - The agent is a callable object (CompiledStateGraph).
  - Custom checkpointer instances are accepted.
  - The agent has the expected public interface (.invoke, .stream).
"""

from unittest.mock import MagicMock

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.tools.system import get_system_time


class TestBuildAgent:
    """build_agent should return a compiled LangGraph agent."""

    def test_returns_callable_agent(self) -> None:
        """The agent should be a callable (CompiledStateGraph) with .invoke and .stream."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        agent = build_agent(llm, [get_system_time])
        assert agent is not None
        assert hasattr(agent, "invoke")
        assert hasattr(agent, "stream")

    def test_accepts_custom_checkpointer(self) -> None:
        """A custom MemorySaver should be used when provided."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        checkpointer = MemorySaver()
        agent = build_agent(llm, [get_system_time], checkpointer=checkpointer)
        assert agent is not None
        assert hasattr(agent, "invoke")

    def test_accepts_system_prompt(self) -> None:
        """A system prompt should be accepted and applied."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        agent = build_agent(
            llm,
            [get_system_time],
            system_prompt="You are a helpful assistant.",
        )
        assert agent is not None

    def test_agent_name_is_set(self) -> None:
        """The compiled agent should have a name attribute."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        agent = build_agent(llm, [get_system_time])
        assert hasattr(agent, "name")

    def test_build_with_multiple_tools(self) -> None:
        """Agent should accept the full tool suite."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        from gem_agentic_ai.tools import web_search, calculate_multiplication

        agent = build_agent(
            llm,
            [get_system_time, calculate_multiplication, web_search],
        )
        assert agent is not None
