"""
TESTS: CONFIGURATION
====================
Verifies that pydantic-settings correctly:
  - Loads default values when env vars are absent.
  - Overrides defaults when env vars are present.
  - Raises ValidationError when the required OPENAI_API_KEY is missing.

These tests ensure the application fails fast on misconfiguration
rather than producing cryptic errors at runtime.
"""

import pytest
from pydantic import ValidationError

from gem_agentic_ai.config import Settings


class TestSettingsDefaults:
    """Fields with defaults should use their default when no env var is set."""

    def test_default_openai_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        s = Settings()
        assert s.openai_model == "gpt-4o-mini"

    def test_default_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("TEMPERATURE", raising=False)
        s = Settings()
        assert s.temperature == 0.0

    def test_default_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        s = Settings()
        assert s.log_level == "INFO"


class TestSettingsCustom:
    """Env vars should override defaults correctly, including type coercion."""

    def test_custom_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.openai_model == "gpt-4"
        assert s.log_level == "DEBUG"

    def test_custom_float_coercion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """pydantic-settings should coerce the string '0.5' to the float 0.5."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("TEMPERATURE", "0.5")
        s = Settings()
        assert s.temperature == 0.5
        assert isinstance(s.temperature, float)


class TestSettingsValidation:
    """Required fields without defaults should raise on construction."""

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When OPENAI_API_KEY is absent and no .env file provides it,
        pydantic-settings should raise ValidationError on construction.

        We use a local subclass that disables .env file loading to
        isolate the test from any .env file on disk.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from pydantic_settings import SettingsConfigDict

        class _TestSettings(Settings):
            model_config = SettingsConfigDict(env_file=None)

        with pytest.raises(ValidationError):
            _TestSettings()
