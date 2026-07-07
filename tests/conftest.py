"""
TEST FIXTURES
=============
Shared pytest fixtures used across the test suite.

WHY fixtures?
  Fixtures provide reusable, composable test resources with
  automatic setup and teardown. They replace setUp/tearDown methods
  from unittest with a more modular approach — each fixture handles
  one concern (e.g., mock env vars, tool instances).
"""

import pytest


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure OPENAI_API_KEY is always set during tests.

    This fixture runs automatically (autouse=True) for every test.
    Without it, importing gem_agentic_ai.config would raise a
    ValidationError because the API key is required.

    Tests that need specific env values should override with their
    own monkeypatch.setenv() calls.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fixture-key")
