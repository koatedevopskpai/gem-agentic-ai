"""
EMPLOYEE DATABASE TOOL
======================
An in-memory SQLite database queried through a LangChain tool.

WHY an in-memory database?
  Zero setup for portfolio reviewers — no Docker, no Postgres, no
  migration scripts. The database is created from scratch every time
  the module is imported, with a small set of sample employees.

  In a production system, this module would be replaced with a
  connection to a real database (PostgreSQL, MySQL, etc.). The tool
  interface (@tool def query_employee_database) remains identical —
  dependency inversion at work.

SECURITY NOTE
  The tool explicitly rejects non-SELECT queries to prevent
  accidental or malicious data modification. In production, this
  would be enforced by database permissions and a read-only
  connection, not just application-level filtering.
"""

import logging
import sqlite3

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Database initialisation ────────────────────────────────────
# Using check_same_thread=False allows reuse across async contexts.
# For this single-threaded demo it is not strictly necessary, but
# including it signals awareness of concurrency considerations.

_conn = sqlite3.connect(":memory:", check_same_thread=False)
_conn.execute(
    "CREATE TABLE employees ("
    "  id INTEGER PRIMARY KEY,"
    "  name TEXT NOT NULL,"
    "  department TEXT NOT NULL,"
    "  location TEXT NOT NULL"
    ")"
)

_conn.executemany(
    "INSERT INTO employees (id, name, department, location) VALUES (?, ?, ?, ?)",
    [
        (1, "Alice Smith", "Engineering", "New York"),
        (2, "Bob Jones", "Marketing", "London"),
        (3, "Charlie Brown", "HR", "Berlin"),
    ],
)
_conn.commit()

logger.info("Employee database initialised with %d records in :memory:.", 3)


@tool
def query_employee_database(sql_query: str) -> str:
    """Run a SQL query against the employee database and return results.

    The database contains a single table with sample employee records:

        employees (
            id         INTEGER   -- unique employee identifier
            name       TEXT      -- full name
            department TEXT      -- department name
            location   TEXT      -- office city
        )

    Sample data:
        1 | Alice Smith    | Engineering | New York
        2 | Bob Jones      | Marketing   | London
        3 | Charlie Brown  | HR          | Berlin

    SAFETY: Only SELECT queries are allowed. The tool inspects the
    query before execution and rejects anything that looks like a
    mutation (INSERT, UPDATE, DELETE, DROP, ALTER, etc.).

    Args:
        sql_query: A SQL SELECT statement.

    Returns:
        Formatted query results, or an error/empty message.

    Example:
        Input:  "SELECT name, location FROM employees WHERE department = 'Engineering'"
        Output: "Alice Smith, New York"
    """
    logger.info("Database query: %s", sql_query)

    # ── Security guard: reject non-SELECT queries ─────────────
    stripped = sql_query.strip().upper()
    if not stripped.startswith("SELECT"):
        logger.warning("Blocked non-SELECT query: %s", sql_query)
        return "Error: Only SELECT queries are permitted for safety."

    # ── Execute query ─────────────────────────────────────────
    try:
        cursor = _conn.execute(sql_query)
        rows = cursor.fetchall()
    except Exception as e:
        logger.exception("SQL query failed: %s", e)
        return f"Query error: {e}"

    # ── Format results ────────────────────────────────────────
    if not rows:
        return "No results found."

    # Each row is a tuple of column values. We join them with
    # commas for a clean, table-like output.
    return "\n".join(", ".join(map(str, row)) for row in rows)
