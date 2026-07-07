"""
CONFIGURATION MODULE
====================
Centralised, type-safe configuration using pydantic-settings.

WHY pydantic-settings instead of os.getenv?
  - Type coercion: TEMPERATURE=0.5 becomes float 0.5, not the string "0.5".
  - Fail-fast: Missing required variables raise immediately at import time,
    not halfway through execution when the LLM is first called.
  - Self-documenting: The Settings class serves as a single source of truth
    for every configuration key the application needs.
  - .env file support: Local development uses .env; production can set
    real environment variables — same code, no changes.

USAGE
  from gem_agentic_ai.config import settings
  llm = ChatOpenAI(model=settings.openai_model, temperature=settings.temperature)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings sourced from environment variables / .env file.

    Every field maps to an environment variable by name convention:
      Field name: openai_api_key   →  Env var: OPENAI_API_KEY
      Field name: log_level        →  Env var: LOG_LEVEL

    Fields without defaults are required — Settings() will raise
    pydantic.ValidationError if they are missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Required ──────────────────────────────────────────────
    openai_api_key: str
    """OpenAI API key. Obtain from https://platform.openai.com/api-keys."""

    # ── Optional (sensible defaults provided) ─────────────────
    openai_model: str = "gpt-4o-mini"
    """LLM model identifier passed to ChatOpenAI. Change to 'gpt-4' for higher quality."""

    temperature: float = 0.0
    """Sampling temperature (0.0–2.0). Lower = more deterministic, higher = more creative."""

    log_level: str = "INFO"
    """Standard Python log level: DEBUG | INFO | WARNING | ERROR | CRITICAL."""


# ── Singleton ───────────────────────────────────────────────────
# Instantiated once at import time. Every module that needs
# configuration imports this single instance rather than
# constructing its own Settings object.
settings = Settings()
