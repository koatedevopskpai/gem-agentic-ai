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

**Context:** The primary agent implementation needs multi-turn conversation, streaming, inspectable control flow, and room to extend the graph (sub-graphs, branching).

**Decision:** Use LangGraph's `create_agent` factory with `MemorySaver` checkpointer.

**Rationale:**
- **Formal state machine:** Nodes and edges are explicitly defined, making control flow inspectable and visualisable.
- **Persistence:** `MemorySaver` checkpoints state after every step. It is in-process only; multi-turn chat survives a restart only once a durable checkpointer is supplied.
- **Streaming:** Agents yield intermediate states, enabling real-time UI updates (tool call progress, typewriter text).
- **Extensibility:** Sub-graphs, parallel tool execution, and branching are configuration changes, not rewrites.
- **Widely adopted:** LangGraph is a commonly used framework for stateful LLM agents.

**Trade-offs:**
- Additional dependency (`langgraph`).
- More abstraction than a simple loop — harder to debug at the lowest level.

**Note:** A manual ReAct loop (`agent/simple.py`) is maintained alongside the LangGraph version as a pedagogical tool, showing exactly what LangGraph does under the hood. It shares the tool library and the execution guard with the graph path (ADR-11).

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
- A scripted chat model (`tests/fakes.py`) allows testing agent loop logic (tool calls, error handling, turn limits) without API calls. Unlike a bare `MagicMock`, it exercises the real message and tool-call shapes.
- `MagicMock` is still used where the test only needs a collaborator that records calls (for example the CLI mode functions).
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
- The simple loop shows exactly what a ReAct agent does: LLM thinks → acts (calls tools) → observes (reads results) → repeats.
- The LangGraph version shows how the same pattern is formalised with a state machine, persistence, and streaming.
- Both share the same tool library and the same `ToolGuard` — demonstrating dependency inversion and, after A03, one execution policy rather than two.
- Portfolio reviewers can trace the evolution from the manual pattern to a framework-based architecture.
- The simple loop is also useful as a debugging tool — if something goes wrong in LangGraph, strip it back to the manual loop and isolate the issue.

---

## ADR-10: Dependency Injection over Hard-Coded Tools

**Status:** Accepted

**Context:** Both agent implementations need access to tools, but should not know about specific tool implementations.

**Decision:** Agents receive tools as parameters (either as `list[BaseTool]` for LangGraph or `Mapping[str, BaseTool]` for the simple loop). Neither agent imports any concrete tool. The registry is also the tool allowlist: a name outside it is refused before execution.

**Rationale:**
- **Testability:** Agents can be tested with mock tools without importing real tools.
- **Swapability:** Adding, removing, or replacing tools requires no changes to agent code.
- **Separation of concerns:** Tools module = "what the agent can do"; agent module = "how the agent decides what to do".
- Follows the dependency inversion principle (D from SOLID).

---

## ADR-11: One Shared Guard for Bounded Tool Execution

**Status:** Accepted

**Context:** A01 pinned a real divergence: an unexpected tool exception became an observation in the manual loop but ended the LangGraph run. Neither path validated arguments against a schema, bounded a slow tool, capped the number of calls, protected a side effect, or exposed why a run stopped. A claim of bounded execution cannot coexist with two different failure behaviours.

**Decision:** Put every tool call on both paths through `ToolGuard`, backed by `ExecutionBudget`:

- `ToolGuard.execute()` validates the call (allowlist, mapping, size, Pydantic schema), reserves budget, checks the idempotency store, enforces approval for consequential tools, and only then runs the tool with a timeout and retry policy.
- The manual loop calls `guard.execute()` directly and turns the `ToolOutcome` into a `ToolMessage`.
- `build_agent()` wraps each tool in `GuardedTool`; the LangGraph tool node receives a finished `ToolMessage`, including `status="error"`, and passes it through unchanged.
- `ExecutionBudget` records the first stop reason and an audit trail of every call.

**Rationale:**
- **One policy:** Validation, retries, idempotency and approval cannot drift between implementations if there is only one implementation.
- **Fail closed:** A tool with no schema, an invalid argument, an unapproved consequential action, or an exhausted budget is refused. It is not executed and then reported.
- **Observable:** `AgentRunResult.budget` and `BoundedAgent.budget` expose the stop reason and per-call records. A later model answer cannot erase an earlier refusal.
- **Recovery:** An unexpected exception becomes an error observation on both paths, so the model can choose another approach.

**Rejected alternatives:**
- **LangChain `ToolCallLimitMiddleware` / `ToolRetryMiddleware`:** They cover the graph path only. The manual loop would keep a second policy, which is how the A01 divergence arose. They also do not provide schema refusal, approval, or idempotency.
- **Catch the exception in the graph and document the divergence:** Rejected. A03 claims bounded execution and a shared failure behaviour; documenting an unexplained difference would leave the claim false on one path.
- **Per-tool try/except in each tool:** Scatters policy across every tool and makes "is this call allowed?" unanswerable in one place.

**Trade-offs and boundaries:**
- The per-call timeout stops the guard waiting; it cannot kill a Python thread. A non-cooperative tool may continue in the background. True isolation needs a process boundary, which is not part of this task.
- The default idempotency store is in-process. `SqliteIdempotencyStore` makes records survive a restart, but it is not a distributed lock; SQLite's primary key plus `INSERT OR IGNORE` gives first-writer-wins.
- Approval is explicit state (`ApprovalStore`), not an interactive graph interrupt. A caller approves an action key before it runs, or between runs.
- Model latency and provider spend are outside the guard. The observation-token cap is an estimated context bound, not a billing measure.
- `GuardedTool.invoke()` returns a `ToolMessage` when given a LangGraph tool-call dict and a string otherwise. That is the protocol the tool node expects, but it is a deliberate change from a plain `BaseTool` and is covered by execution tests.

---

## ADR-12: Durable Memory with an Enforced User Boundary

**Status:** Accepted

**Context:** `MemorySaver` keeps checkpoints in a dictionary, so a conversation ends with the process. It is also keyed only by the caller's `thread_id`: nothing verifies which user a thread belongs to. A service that accepts a thread id from a request could therefore resume a foreign conversation. Durability and scoping have to be solved together, because durable storage makes an ownership mistake persistent.

**Decision:** Add `DurableMemory` (persistence.py), which combines three pieces:

- `SqliteSaver` over a local SQLite file stores LangGraph checkpoints.
- A `thread_registry` table maps `(user_id, thread_key)` to a random, opaque storage id, with creation and last-use times.
- `_ScopedCheckpointer` wraps the saver and authorises every `get_tuple`, `list`, `put` and `put_writes` against that registry. A config missing either value, or naming a pair that is not registered, raises `ThreadAccessError`.

Deletion is only available as `DurableMemory.delete_thread(user_id, thread_key)`, because the saver's `delete_thread(thread_id)` has no user context. Retention is an explicit `sweep_expired()` over `last_used_at`; opening the store never deletes data. `memory.guard()` supplies a `ToolGuard` whose SQLite idempotency records sit beside the checkpoints, so a replayed step does not repeat a completed action.

**Rationale:**
- **Fail closed:** The graph cannot read or write a thread whose config does not carry a registered pair. A forgotten call-site check cannot expose a conversation.
- **Opaque ids:** Knowing another user's thread key does not yield a usable storage id.
- **Durability where it is needed:** Checkpoints and completed actions survive a restart; approval decisions deliberately do not, because they record a human decision rather than conversation state.
- **Local and free:** One SQLite file, no service, no credentials. The scoped wrapper is not tied to SQLite in principle.
- **Observable lifecycle:** Listing, deletion and retention return records rather than acting silently.

**Rejected alternatives:**
- **Embed the user in the thread id (`alice:support-1`):** Any caller can construct one, and it leaks the user id into logs and keys. Nothing verifies the prefix.
- **Enforce ownership only at the call site:** One missed check resumes a foreign conversation, and durability makes the mistake permanent.
- **A JSON file per thread:** No atomic multi-key writes or ordered pending writes; rebuilding what the SQLite saver already provides is wasted effort.
- **Auto-sweep on open:** Convenient and surprising. Deletion should be schedulable and auditable.

**Trade-offs and boundaries:**
- One SQLite file on one host. This is not a distributed store or a cross-host lock.
- Ownership is only as strong as the `user_id` the application supplies; the checkpointer cannot authenticate a user.
- The directory is not encrypted. Filesystem access bypasses the boundary.
- The sync saver has no async implementation; `ainvoke` / `astream` are unsupported with this checkpointer at this version.
- A crash after an external side effect but before the idempotency write repeats the action on replay. The store narrows the window; it does not close the dual-write gap, and tools without a stable action key remain at-least-once. `docs/A02.md` lists what is restart-safe and what needs reconciliation.

---

## Component Dependency Graph

```
main.py
  ├── config.py          (pydantic-settings)
  ├── tools/__init__.py  (tool registry)
  │   ├── tools/system.py
  │   ├── tools/web.py
  │   ├── tools/database.py
  │   └── tools/actions.py   (consequential demo action)
  └── agent/__init__.py  (agent registry)
      ├── agent/simple.py  ← uses tools dict + guard
      └── agent/graph.py   ← uses tools list + models/state.py + guard

guard.py
  ├── bounds.py          (limits, budget, stop reasons)
  └── idempotency.py     (in-memory and SQLite stores)

persistence.py
  ├── langgraph.checkpoint.sqlite  (SqliteSaver)
  ├── bounds.py                    (ToolLimits for guard())
  ├── guard.py                     (ToolGuard)
  └── idempotency.py               (SqliteIdempotencyStore)
```

Key property: **arrows point in the direction of dependency.** Tools never import agents. Agents never import specific tools. Configuration is imported by everything but imports nothing (except pydantic-settings).
