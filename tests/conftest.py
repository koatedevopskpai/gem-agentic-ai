"""
TEST FIXTURES
=============
Shared pytest fixtures used across the test suite.

WHY seed the environment at import time?
  ``gem_agentic_ai.config`` instantiates ``Settings()`` at module import,
  and ``OPENAI_API_KEY`` is a required field. A fixture alone is too late:
  pytest collects (and therefore imports) test modules before it runs any
  fixture, so on a clean checkout with no ``.env`` file collection fails
  with a ValidationError.

  pytest imports ``conftest.py`` before collecting test modules, so setting
  the variable at module level makes the suite runnable without credentials
  or a local ``.env``. No test makes a live model call: every agent test
  scripts or mocks the LLM, and the placeholder key is only there to let
  ``ChatOpenAI`` construct.

WHY fixtures?
  Fixtures provide reusable, composable test resources with
  automatic setup and teardown. They replace setUp/tearDown methods
  from unittest with a more modular approach — each fixture handles
  one concern (e.g. mock env vars, tool instances).
"""

import os

import pytest

# Must run before any test module imports gem_agentic_ai.config.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fixture-key")


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin OPENAI_API_KEY for every test and restore it at teardown.

    The module-level ``setdefault`` above protects collection; this fixture
    keeps the value stable if an individual test changes it.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fixture-key")
