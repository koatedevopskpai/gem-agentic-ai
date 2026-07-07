"""
TESTS: CLI ENTRY POINT
======================
Tests argument parsing, shared helpers, and mode dispatch.

We test parse_args and helper functions in isolation (no LLM calls).
The full main() function is not tested here because it constructs
agents that require an API key — that is an integration concern.
"""

from unittest.mock import MagicMock, patch

import pytest

from gem_agentic_ai.main import (
    _build_llm,
    _extract_final_answer,
    _get_tools,
    _get_tools_by_name,
    parse_args,
)
from gem_agentic_ai.config import settings


# ======================================================================
# ARGUMENT PARSING
# ======================================================================


class TestParseArgs:
    """Tests for the argparse-based CLI argument parser."""

    def test_default_mode_is_demo(self) -> None:
        """Running with no arguments should default to 'demo' mode."""
        args = parse_args([])
        assert args.mode == "demo"
        assert args.verbose is False

    def test_demo_mode_explicit(self) -> None:
        args = parse_args(["demo"])
        assert args.mode == "demo"

    def test_chat_mode(self) -> None:
        args = parse_args(["chat"])
        assert args.mode == "chat"

    def test_simple_mode(self) -> None:
        args = parse_args(["simple"])
        assert args.mode == "simple"

    def test_verbose_flag_long(self) -> None:
        """--verbose should set verbose=True."""
        args = parse_args(["demo", "--verbose"])
        assert args.verbose is True

    def test_verbose_flag_short_form(self) -> None:
        args = parse_args(["demo", "-v"])
        assert args.verbose is True

    def test_invalid_mode_raises_system_exit(self) -> None:
        """An unrecognised mode should cause argparse to exit with an error."""
        with pytest.raises(SystemExit):
            parse_args(["invalid_mode"])

    def test_help_flag_exits_cleanly(self) -> None:
        """--help should exit cleanly (SystemExit with code 0)."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0


# ======================================================================
# SHARED HELPERS
# ======================================================================


class TestBuildLlm:
    """_build_llm should create a ChatOpenAI from configured settings."""

    def test_returns_chat_openai(self) -> None:
        llm = _build_llm()
        assert llm is not None
        # Verify the model name from settings was used.
        assert "gpt" in llm.model_name.lower()


class TestGetTools:
    """_get_tools should return the full list of registered tools."""

    def test_returns_all_tools(self) -> None:
        tools = _get_tools()
        names = {t.name for t in tools}
        assert names == {
            "get_system_time",
            "calculate_multiplication",
            "web_search",
            "query_employee_database",
        }

    def test_each_tool_is_invocable(self) -> None:
        """Every returned tool should have a working .invoke() method."""
        tools = _get_tools()
        for t in tools:
            assert hasattr(t, "invoke")


class TestGetToolsByName:
    """_get_tools_by_name should build a correct name→tool lookup."""

    def test_all_tools_present(self) -> None:
        lookup = _get_tools_by_name()
        assert "get_system_time" in lookup
        assert "web_search" in lookup
        assert "calculate_multiplication" in lookup
        assert "query_employee_database" in lookup

    def test_values_are_invocable(self) -> None:
        """Each tool should have an .invoke() method (StructuredTool interface)."""
        lookup = _get_tools_by_name()
        for name, tool in lookup.items():
            assert hasattr(tool, "invoke"), f"Tool '{name}' has no .invoke()"

    def test_tool_matches_name(self) -> None:
        """The key and the tool's .name attribute should match."""
        lookup = _get_tools_by_name()
        for name, tool in lookup.items():
            assert tool.name == name


class TestExtractFinalAnswer:
    """_extract_final_answer should pull the last AIMessage.content from events."""

    def _make_event(self, content: str = "", tool_calls: list | None = None):
        """Helper: create a fake LangGraph event with an AIMessage."""
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls or []
        return {"agent": {"messages": [msg]}}

    def test_extracts_content_from_last_message(self) -> None:
        events = [self._make_event(content="Paris")]
        assert _extract_final_answer(events) == "Paris"

    def test_skips_tool_call_events(self) -> None:
        """Events with tool calls should be skipped; only content-bearing events count."""
        events = [
            self._make_event(tool_calls=[{"name": "search", "args": {}}]),
            self._make_event(content="Final answer here."),
        ]
        assert _extract_final_answer(events) == "Final answer here."

    def test_returns_empty_string_when_no_content(self) -> None:
        events = [self._make_event(tool_calls=[{"name": "search", "args": {}}])]
        assert _extract_final_answer(events) == ""

    def test_multiple_content_events_uses_last(self) -> None:
        """Only the LAST content-bearing event should be returned."""
        events = [
            self._make_event(content="First answer"),
            self._make_event(content="Second answer (final)"),
        ]
        assert _extract_final_answer(events) == "Second answer (final)"

    def test_node_without_messages_key_is_ignored(self) -> None:
        events = [
            {"some_node": {"other_data": "no messages here"}},
            self._make_event(content="Actual answer"),
        ]
        assert _extract_final_answer(events) == "Actual answer"
