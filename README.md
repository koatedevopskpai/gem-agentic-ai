# Gem Agentic AI

[![ci](https://github.com/koatedevopskpai/gem-agentic-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/koatedevopskpai/gem-agentic-ai/actions/workflows/ci.yml)

A production-grade LLM agent orchestration framework demonstrating two architectural approaches to the ReAct (Reasoning + Acting) pattern:

1. **Simple ReAct Loop** — a manual thought-action-observation loop (~50 lines) showing the core pattern from first principles.
2. **LangGraph State Machine** — a formal state graph with checkpointing, streaming, and typed state management.

Both implementations consume the same shared tool library, illustrating dependency inversion and the power of abstracting tool interfaces.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     CLI (argparse)                       │
│         python -m gem_agentic_ai {demo|chat|simple}      │
└────────────────────┬────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌─────────┐   ┌───────────┐   ┌───────────┐
│  DEMO   │   │   CHAT    │   │  SIMPLE   │
│(LangGraph│   │(LangGraph │   │(ReActLoop)│
│ single)  │   │ multi-turn│   │ single)   │
└────┬─────┘   └─────┬─────┘   └─────┬─────┘
     │               │               │
     └───────────────┼───────────────┘
                     ▼
          ┌───────────────────┐
          │   TOOL LIBRARY    │
          │                   │
          │  system.py  (✓)   │
          │  web.py     (🌐)   │
          │  database.py(🗄️)   │
          └───────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Project layout | `src/` | Prevents import ambiguity; follows PyPA recommendation |
| Configuration | `pydantic-settings` | Type-safe, validated env vars fail fast at startup |
| Web search | `duckduckgo_search` direct | Avoids deprecated `langchain-community` wrapper |
| Agent orchestration | LangGraph `create_agent` | Formal state machine with persistence and streaming |
| CLI | `argparse` (stdlib) | Zero dependencies; explicit and easy to follow |
| Logging | `logging` module | Structured output with levels; production standard |
| Testing | `pytest` + `pytest-cov` | Offline suite with a scripted model; coverage reported by CI, not gated |

---

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# 2. Install pinned dependencies and the package itself
pip install -r requirements-dev.txt
pip install --no-deps -e .        # src layout: required for `import gem_agentic_ai`

# 3. Configure your API key (only needed for live model runs)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

`requirements.txt` pins the runtime versions used in CI. `pip install -e ".[dev]"`
also works, but it resolves the newest matching versions rather than the
pinned set. See [docs/CLEAN_CHECKOUT.md](docs/CLEAN_CHECKOUT.md) for the
verified baseline.

### Usage

```bash
# Single-query demo (LangGraph agent) — default mode
python -m gem_agentic_ai demo

# Multi-turn conversation with persistent memory
python -m gem_agentic_ai chat

# Single query with the basic ReAct loop (no LangGraph)
python -m gem_agentic_ai simple

# Enable verbose logging
python -m gem_agentic_ai demo --verbose
```

### Example Queries

```
👤 Enter your query: What is the current date and time?
👤 Enter your query: Who is in the Engineering department?
👤 Enter your query: What is 15 multiplied by 27?
👤 Enter your query: Search the web for the latest developments in AI
```

---

## Project Structure

```
src/gem_agentic_ai/
├── __init__.py          # Package marker
├── __main__.py          # python -m entry point
├── config.py            # pydantic-settings configuration
├── main.py              # CLI argument parsing and mode dispatch
├── tools/
│   ├── __init__.py      # Tool registry / re-exports
│   ├── system.py        # Standard-library tools (clock, calculator)
│   ├── web.py           # DuckDuckGo web search
│   └── database.py      # In-memory SQLite employee database
├── agent/
│   ├── __init__.py      # Agent registry / re-exports
│   ├── simple.py        # Manual ReAct loop (~50 lines)
│   └── graph.py         # LangGraph state machine builder
└── models/
    ├── __init__.py      # Model registry / re-exports
    └── state.py         # AgentState TypedDict for LangGraph
```

---

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests with coverage
pytest

# Run only unit tests (skip integration tests)
pytest -m "not integration"

# Run integration tests (hits DuckDuckGo API)
pytest -m integration

# Generate HTML coverage report
pytest --cov-report=html
# Open htmlcov/index.html in your browser
```

---

## Architectural Decision Records

Detailed design rationale for every major decision is documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). This includes trade-offs considered, rejected alternatives, and the reasoning behind each choice.

---

## Two Agent Architectures: Comparison

| Feature | Simple ReAct (`simple.py`) | LangGraph (`graph.py`) |
|---------|---------------------------|----------------------|
| State management | Manual list of messages | Formal state machine |
| Persistence | None (stateless) | MemorySaver (in-memory) |
| Streaming | Blocking (full response) | Event-based streaming |
| Tool routing | Manual `if/else` | Built-in conditional edges |
| Human-in-the-loop | No | Yes (via interrupt) |
| Code complexity | ~50 lines (pedagogical) | ~30 lines (builder) |
| Production readiness | Learning/teaching | Production deployment |

---

## License

This project is provided as a portfolio piece. See LICENSE file for details.
