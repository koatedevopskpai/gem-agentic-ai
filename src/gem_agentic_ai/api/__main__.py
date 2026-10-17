"""Run the API server.

    GEM_API_TOKENS="token:user" python -m gem_agentic_ai.api

Environment:
  GEM_API_TOKENS   token:user pairs, comma separated (required)
  GEM_MEMORY_DIR   directory for the durable SQLite stores (default var/agent-memory)
  GEM_API_HOST     bind address (default 127.0.0.1)
  GEM_API_PORT     port (default 8000)
  OPENAI_API_KEY   provider credential for ChatOpenAI (required)

The server refuses to start without tokens: there is no anonymous mode.
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI
from langchain_openai import ChatOpenAI

from gem_agentic_ai.api.auth import parse_tokens
from gem_agentic_ai.api.server import create_app
from gem_agentic_ai.config import settings
from gem_agentic_ai.main import _get_tools
from gem_agentic_ai.persistence import DurableMemory


def build_app_from_env() -> FastAPI:
    """Construct the application from environment variables, failing closed."""
    tokens = parse_tokens(os.environ.get("GEM_API_TOKENS", ""))
    memory = DurableMemory(os.environ.get("GEM_MEMORY_DIR", "var/agent-memory"))
    llm = ChatOpenAI(model=settings.openai_model, temperature=settings.temperature)
    return create_app(llm=llm, tokens=tokens, memory=memory, tools=_get_tools())


def main() -> None:
    host = os.environ.get("GEM_API_HOST", "127.0.0.1")
    port = int(os.environ.get("GEM_API_PORT", "8000"))
    uvicorn.run(build_app_from_env(), host=host, port=port)


if __name__ == "__main__":
    main()
