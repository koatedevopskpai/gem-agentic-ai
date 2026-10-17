# Architectural Decision Records (ADRs)

This document captures the key architectural decisions made during the design of Gem Agentic AI. Each ADR describes the context, the decision, the rationale, and the trade-offs considered.

---

## ADR-1: Project Layout — `src/` Package Structure

**Status:** Accepted

**Context:** The project needed a scalable directory structure that would prevent import ambiguity, support testing, and signal production intent.

**Decision:** Use the `src/` layout with `src/gem_agentic_ai/` as the importable package root.

**Rationale:**
- Prevents accidental imports of local modules instead of installed packages (a common Python pitfall).
- Follows the Python Packaging Authority's recommendation for modern projects.
- When `pip install -e .` is used, the package is installed from `src/`, making imports unambiguous.
- Tests run against the installed package, not the loose source files — catching packaging issues early.

**Trade-offs:**
- Slightly more complex project structure than a flat layout.
- Requires an editable install for development (`pip install -e .`).

---

## ADR-2: Configuration Management — pydantic-settings

**Status:** Accepted

**Context:** The application needs to read configuration from environment variables with sensible defaults, type coercion, and fail-fast validation.

**Decision:** Use `pydantic-settings` (v2) with a `Settings` class.

**Rationale:**
- **Type safety:** `TEMPERATURE=0.5` is automatically coerced to `float 0.5`, not the string `"0.5"`.
- **Fail-fast:** Missing required variables (e.g., `OPENAI_API_KEY`) raise `ValidationError` at import time, not when the LLM is first called.
- **Self-documenting:** The `Settings` class serves as a single source of truth for every configuration key.
- **.env support:** Reads from `.env` file automatically with `model_config = SettingsConfigDict(env_file=".env")`.
- **Widely adopted:** The standard for Pydantic-based applications.

**Rejected alternatives:**
- `os.getenv()`: No type coercion, no validation, scattered calls across the codebase.
- `python-dotenv` alone: Just loads env vars, no validation or typing.
- `python-decouple`: Less integrated with modern Python typing patterns.

---

## ADR-3: Web Search — Direct duckduckgo_search over langchain-community

**Status:** Accepted

**Context:** The agent needs web search capability. LangChain-community provides `DuckDuckGoSearchRun`, but the package is deprecated and emits runtime warnings.

**Decision:** Use the `duckduckgo_search` library directly, wrapping it in a `@tool` decorator.

**Rationale:**
- Eliminates deprecation warnings from `langchain-community`.
- Reduces dependency surface area (one fewer LangChain integration package).
- The `@tool` decorator makes the function a first-class LangChain tool regardless of backend.
- DuckDuckGo requires no API key — ideal for demos and portfolio review.
- Lazy import (`from duckduckgo_search import DDGS` inside the function body) ensures that a missing library only breaks the search tool, not the entire agent.

**Rejected alternatives:**
- `langchain-community.DuckDuckGoSearchRun`: Functionally identical, but emits deprecation warnings and adds an unnecessary dependency.
- SerpAPI / Tavily: Require API keys and paid accounts — inappropriate for a portfolio piece meant to be run without sign-up.

---

## ADR-4: Agent Orchestration — LangGraph over Manual Loop

**Status:** Accepted

**Context:** The primary agent implementation needs to be production-grade — supporting multi-turn conversation, streaming, and future extensibility (human-in-the-loop, branching).

**Decision:** Use LangGraph's `create_agent` factory with `MemorySaver` checkpointer.

**Rationale:**
- **Formal state machine:** Nodes and edges are explicitly defined, making control flow inspectable and visualisable.
- **Persistence:** `MemorySaver` checkpoints state after every step. Multi-turn chat survives if a database-backed checkpointer is swapped in.
- **Streaming:** Agents yield intermediate states, enabling real-time UI updates (tool call progress, typewriter text).
- **Extensibility:** Human-in-the-loop, sub-graphs, parallel tool execution, and branching are configuration changes, not rewrites.
- **Industry standard:** LangGraph is the leading framework for production LLM agents.

**Trade-offs:**
- Additional dependency (`langgraph`).
- More abstraction than a simple loop — harder to debug at the lowest level.

**Note:** A manual ReAct loop (`agent/simple.py`) is maintained alongside the LangGraph version as a pedagogical tool, showing exactly what LangGraph does under the hood.

---

## ADR-5: CLI Framework — argparse

**Status:** Accepted

**Context:** The application needs a command-line interface with two operation modes (demo/chat/simple) and a verbose flag.

**Decision:** Use `argparse` from the Python standard library.

**Rationale:**
- Zero dependencies — important when every dependency is justified.
- Explicit and easy for non-technical readers to follow (no decorator magic).
- Sufficient for the application's needs (two modes, one optional flag).
- Built-in `--help` generation.

**Rejected alternatives:**
- `click`/`typer`: More features (subcommands, autocomplete, colours) but add dependencies that are unnecessary for a two-command CLI.
- `fire`: Too magical — auto-generates CLIs from function signatures, obscuring the interface.

---

## ADR-6: Logging — Standard Library `logging`

**Status:** Accepted

**Context:** The application needs structured output with severity levels, configurability, and standard production tooling compatibility.

**Decision:** Use Python's built-in `logging` module.

**Rationale:**
- Zero dependencies.
- Log levels (DEBUG, INFO, WARNING, ERROR) provide appropriate granularity.
- Consistent format including timestamp, level, module name, and message.
- Output can be redirected to files, external aggregators, or suppressed entirely.
- The log level is configurable via environment variable, with `--verbose` flag override at runtime.
- Standard tool for all production Python applications.

**Rejected alternatives:**
- `print()`: No levels, no structured output, no configurability.
- `structlog`: Powerful but adds a dependency for a simple use case.
- `loguru`: Excellent DX but not standard-library; adds a dependency.

---

## ADR-7: Database — In-Memory SQLite

**Status:** Accepted

**Context:** The demo needs a queryable database to demonstrate the database tool. The solution must require zero setup for portfolio reviewers.

**Decision:** Use `sqlite3` with `:memory:` database.

**Rationale:**
- Zero setup — no Docker, no Postgres, no migration scripts.
- Created and seeded at module import time with sample employee data.
- The `@tool` interface abstracts the storage backend — swapping to PostgreSQL would require no changes to the agent or CLI.
- `check_same_thread=False` demonstrates concurrency awareness.
- A security guard rejects non-SELECT queries, preventing accidental data modification.

**Production path:** Replace the in-memory connection with a connection to an external database (PostgreSQL, MySQL, etc.). The tool signature remains identical.

---

## ADR-8: Testing — pytest with Fixtures and Mocking

**Status:** Accepted

**Context:** The test suite must verify agent orchestration logic, tool correctness, CLI parsing, and configuration validation without requiring an LLM API key.

**Decision:** Use `pytest` with `pytest-cov` for coverage, `monkeypatch` for environment variable control, and `unittest.mock` for agent/tool mocking.

**Rationale:**
- Industry standard testing framework.
- `monkeypatch` fixtures enable per-test environment variable control without cross-test contamination.
- Mocking the LLM (`MagicMock`) allows testing agent loop logic (tool calls, error handling, turn limits) without API calls.
- `sys.modules` manipulation enables testing lazy import error paths.
- `conftest.py` sets a placeholder `OPENAI_API_KEY` at import time. This must happen before collection: `config.py` instantiates `Settings()` at import, and a fixture would run too late on a clean checkout with no `.env`. The autouse fixture then pins the value per test.
- Coverage is reported, not asserted. No coverage threshold is enforced, so a high percentage is not evidence that the suite is comprehensive.
- Integration test marker (`@pytest.mark.integration`) separates fast unit tests from network-dependent tests. The default run is `-m "not integration"`.

---

## ADR-9: Keeping Both Agent Implementations

**Status:** Accepted

**Context:** The portfolio needs to demonstrate architectural progression, not just the final state.

**Decision:** Keep and maintain both `agent/simple.py` (manual ReAct loop) and `agent/graph.py` (LangGraph state machine).

**Rationale:**
- The simple loop (~50 lines) shows exactly what a ReAct agent does: LLM thinks → acts (calls tools) → observes (reads results) → repeats.
- The LangGraph version shows how the same pattern is formalised with a state machine, persistence, and streaming.
- Both share the same tool library — demonstrating dependency inversion and code reuse.
- Portfolio reviewers can trace the evolution from basic pattern to production architecture.
- The simple loop is also useful as a debugging tool — if something goes wrong in LangGraph, strip it back to the manual loop and isolate the issue.

---

## ADR-10: Dependency Injection over Hard-Coded Tools

**Status:** Accepted

**Context:** Both agent implementations need access to tools, but should not know about specific tool implementations.

**Decision:** Agents receive tools as parameters (either as `list[BaseTool]` for LangGraph or `dict[str, object]` for the simple loop). Neither agent imports any concrete tool.

**Rationale:**
- **Testability:** Agents can be tested with mock tools without importing real tools.
- **Swapability:** Adding, removing, or replacing tools requires no changes to agent code.
- **Separation of concerns:** Tools module = "what the agent can do"; agent module = "how the agent decides what to do".
- Follows the dependency inversion principle (D from SOLID).

---

## Component Dependency Graph

```
main.py
  ├── config.py          (pydantic-settings)
  ├── tools/__init__.py  (tool registry)
  │   ├── tools/system.py
  │   ├── tools/web.py
  │   └── tools/database.py
  └── agent/__init__.py  (agent registry)
      ├── agent/simple.py  ← uses tools dict
      └── agent/graph.py   ← uses tools list + models/state.py
```

Key property: **arrows point in the direction of dependency.** Tools never import agents. Agents never import specific tools. Configuration is imported by everything but imports nothing (except pydantic-settings).
