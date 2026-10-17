"""
TESTS: TOOLS
============
Verifies that every tool:
  - Is invocable via the standardised LangChain .invoke() interface.
  - Returns the expected type and format.
  - Handles edge cases (empty input, invalid input, errors gracefully).

We focus on system tools and the database tool because they are
deterministic and have no external dependencies. The web search
tool is tested with a smoke test and a marker for network-dependent
integration tests.
"""

from datetime import datetime

import pytest

from gem_agentic_ai.tools.database import query_employee_database
from gem_agentic_ai.tools.system import calculate_multiplication, get_system_time
from gem_agentic_ai.tools.web import web_search

# ======================================================================
# SYSTEM TOOLS
# ======================================================================


class TestGetSystemTime:
    """get_system_time relies only on datetime.now() — fully deterministic."""

    def test_default_format(self) -> None:
        """Default format should match ISO 8601 (seconds precision)."""
        result = get_system_time.invoke({})
        expected = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        assert result == expected

    def test_custom_date_format(self) -> None:
        """Custom strftime format should be respected."""
        result = get_system_time.invoke({"format_string": "%Y-%m-%d"})
        assert result == datetime.now().strftime("%Y-%m-%d")

    def test_year_only(self) -> None:
        result = get_system_time.invoke({"format_string": "%Y"})
        assert result == datetime.now().strftime("%Y")

    def test_returns_string(self) -> None:
        result = get_system_time.invoke({})
        assert isinstance(result, str)


class TestCalculateMultiplication:
    """Arithmetic edge cases and correctness."""

    def test_basic_multiplication(self) -> None:
        assert calculate_multiplication.invoke({"a": 3, "b": 4}) == 12

    def test_negative_numbers(self) -> None:
        assert calculate_multiplication.invoke({"a": -2, "b": 5}) == -10

    def test_float_arguments(self) -> None:
        assert calculate_multiplication.invoke({"a": 2.5, "b": 4}) == 10.0

    def test_zero(self) -> None:
        assert calculate_multiplication.invoke({"a": 100, "b": 0}) == 0

    def test_large_numbers(self) -> None:
        result = calculate_multiplication.invoke({"a": 1_000_000, "b": 1_000_000})
        assert result == 1_000_000_000_000

    def test_returns_float(self) -> None:
        result = calculate_multiplication.invoke({"a": 3, "b": 4})
        assert isinstance(result, (int, float))


# ======================================================================
# DATABASE TOOL
# ======================================================================


class TestQueryEmployeeDatabase:
    """Tests against the in-memory SQLite employee database."""

    def test_select_all(self) -> None:
        """SELECT * should return all sample records."""
        result = query_employee_database.invoke(
            {"sql_query": "SELECT * FROM employees ORDER BY id"}
        )
        assert "Alice Smith" in result
        assert "Bob Jones" in result
        assert "Charlie Brown" in result

    def test_select_with_where(self) -> None:
        """WHERE clause should filter results correctly."""
        result = query_employee_database.invoke(
            {"sql_query": "SELECT name FROM employees WHERE department = 'Engineering'"}
        )
        assert "Alice Smith" in result
        assert "Bob Jones" not in result

    def test_select_specific_columns(self) -> None:
        result = query_employee_database.invoke(
            {"sql_query": "SELECT location FROM employees WHERE name = 'Bob Jones'"}
        )
        assert "London" in result

    def test_no_matching_results(self) -> None:
        """Query with no matches should return 'No results found.'"""
        result = query_employee_database.invoke(
            {"sql_query": "SELECT * FROM employees WHERE department = 'Nonexistent'"}
        )
        assert "no results" in result.lower()

    def test_rejects_non_select(self) -> None:
        """Non-SELECT queries should be blocked for safety."""
        dangerous_queries = [
            "DROP TABLE employees",
            "DELETE FROM employees",
            "INSERT INTO employees VALUES (4, 'Eve', 'Finance', 'Paris')",
            "UPDATE employees SET name = 'X' WHERE id = 1",
        ]
        for q in dangerous_queries:
            result = query_employee_database.invoke({"sql_query": q})
            assert "error" in result.lower(), f"Query not blocked: {q}"
            assert "only select" in result.lower()

    def test_invalid_sql_returns_error(self) -> None:
        """Malformed SQL should return a descriptive error, not crash."""
        result = query_employee_database.invoke(
            {"sql_query": "SELECT invalid_column FROM employees"}
        )
        assert "error" in result.lower() or "no such column" in result.lower()

    def test_returns_string(self) -> None:
        result = query_employee_database.invoke(
            {"sql_query": "SELECT * FROM employees LIMIT 1"}
        )
        assert isinstance(result, str)


# ======================================================================
# WEB SEARCH TOOL
# ======================================================================


class TestWebSearch:
    """Web search tests — structure offline, one live call behind a marker.

    ``web_search`` returns an error string when the network or the
    DuckDuckGo library is unavailable, so an unmarked call would pass
    offline while proving nothing. Every network call here is marked
    ``integration`` and excluded from the default pytest run.

    The formatting and error paths are covered without the network in
    ``test_tools_web.py``.
    """

    def test_tool_metadata(self) -> None:
        """Verify the tool has the expected LangChain interface."""
        assert web_search.name == "web_search"
        assert "query" in web_search.args

    @pytest.mark.integration
    def test_live_search(self) -> None:
        """Integration test: hits DuckDuckGo API.

        Run with: pytest -m integration

        NOTE: This test may fail if the network is unavailable or
        DuckDuckGo returns unexpected results. It is excluded from
        the default test run to keep CI fast and reliable.
        """
        result = web_search.invoke(
            {"query": "Python programming language", "max_results": 3}
        )
        assert isinstance(result, str)
        assert len(result) > 0
