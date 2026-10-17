"""
TESTS: MAIN ENTRY POINT (DISPATCH & LOGGING)
=============================================
Tests the main() dispatcher and _configure_logging() helper that
are not covered by the unit tests for individual helper functions.
"""

import logging
from unittest.mock import MagicMock, patch

from gem_agentic_ai.bounds import AgentRunResult, BudgetSnapshot, StopReason
from gem_agentic_ai.config import settings
from gem_agentic_ai.main import _configure_logging, main


def _run_result(answer: str) -> AgentRunResult:
    """Build the value run_simple_agent returns, without running the model."""
    return AgentRunResult(
        answer=answer,
        stop_reason=StopReason.COMPLETED,
        budget=BudgetSnapshot(
            stop_reason=None,
            calls=0,
            per_tool={},
            observation_tokens=0,
            elapsed_seconds=0.0,
            cancelled=False,
            records=(),
        ),
    )


class TestConfigureLogging:
    """_configure_logging should set up the root logger correctly."""

    def test_sets_log_level_from_settings(self) -> None:
        _configure_logging()
        root = logging.getLogger()
        # The root logger should be set to DEBUG or INFO.
        assert root.level in (logging.DEBUG, logging.INFO)

    def test_configure_logging_does_not_raise(self) -> None:
        # Calling it twice should be harmless.
        _configure_logging()
        _configure_logging()  # second call


class TestMainDispatch:
    """main() should dispatch to the correct mode function."""

    def test_main_does_not_raise(self) -> None:
        """main() with demo args should run without error.

        We pass the mode via argv, and the mode function is mocked
        to avoid actual LLM calls.
        """
        with patch("gem_agentic_ai.main.run_demo") as mock_demo:
            main(["demo"])
            mock_demo.assert_called_once()

    def test_main_chat_dispatch(self) -> None:
        with patch("gem_agentic_ai.main.run_chat") as mock_chat:
            main(["chat"])
            mock_chat.assert_called_once()

    def test_main_simple_dispatch(self) -> None:
        with patch("gem_agentic_ai.main.run_simple_demo") as mock_simple:
            main(["simple"])
            mock_simple.assert_called_once()

    def test_main_verbose_sets_log_level(self) -> None:
        """--verbose should change log_level to DEBUG before dispatch."""
        original_level = settings.log_level
        try:
            with patch("gem_agentic_ai.main.run_demo"):
                main(["demo", "--verbose"])
            assert settings.log_level == "DEBUG"
        finally:
            settings.log_level = original_level


class TestModeFunctions:
    """Mode functions (run_demo, run_chat, run_simple_demo) with mocked agents."""

    def test_run_demo_completes(self) -> None:
        """run_demo should execute without error when agent is mocked."""
        from gem_agentic_ai.main import run_demo

        with patch("builtins.input", return_value="test query"):
            with patch("gem_agentic_ai.main.build_agent") as mock_build:
                mock_agent = MagicMock()
                # Simulate a LangGraph agent stream returning one event
                # with an AIMessage that has content but no tool calls.
                mock_msg = MagicMock()
                mock_msg.content = "Final answer"
                mock_msg.tool_calls = []
                mock_agent.stream.return_value = [
                    {"agent": {"messages": [mock_msg]}}
                ]
                mock_build.return_value = mock_agent

                run_demo()  # should not raise

    def test_run_simple_demo_completes(self) -> None:
        """run_simple_demo should execute without error when agent is mocked."""
        from gem_agentic_ai.main import run_simple_demo

        with patch("builtins.input", return_value="test query"):
            with patch("gem_agentic_ai.main.run_simple_agent") as mock_run:
                mock_run.return_value = _run_result("Mocked answer")

                run_simple_demo()  # should not raise
                mock_run.assert_called_once()

    def test_run_demo_with_tool_call_event(self) -> None:
        """run_demo should handle tool-call events and still extract the final answer."""
        from gem_agentic_ai.main import run_demo

        with patch("builtins.input", return_value="test"):
            with patch("gem_agentic_ai.main.build_agent") as mock_build:
                # First event: tool call, second event: final answer
                tool_msg = MagicMock()
                tool_msg.content = ""
                tool_msg.tool_calls = [{"name": "search_tool"}]

                answer_msg = MagicMock()
                answer_msg.content = "Computed answer"
                answer_msg.tool_calls = []

                mock_agent = MagicMock()
                mock_agent.stream.return_value = [
                    {"agent": {"messages": [tool_msg]}},
                    {"agent": {"messages": [answer_msg]}},
                ]
                mock_build.return_value = mock_agent

                run_demo()  # should not raise
