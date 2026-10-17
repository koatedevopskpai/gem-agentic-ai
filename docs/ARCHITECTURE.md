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
- **Recovery at a documented boundary:** `GuardedTool` converts an unexpected exception into an error `ToolMessage` before the native tool node sees it, so the model can choose another approach. LangGraph's own exception behaviour is unchanged; it is bypassed by the wrapper, not modified. A graph built without the wrapper keeps the native behaviour.

**Rejected alternatives:**
- **LangChain `ToolCallLimitMiddleware` / `ToolRetryMiddleware`:** They cover the graph path only. The manual loop would keep a second policy, which is how the A01 divergence arose. They also do not provide schema refusal, approval, or idempotency.
- **Catch the exception in the graph and stop there:** Rejected. That would leave the manual loop with a different validation, retry and approval policy, and would couple the fix to the native tool node instead of putting one wrapper around it.
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

## ADR-13: An Isolated Runner for One Bounded Coding Task

**Status:** Accepted

**Context:** ADR-11 records the limit of an in-process execution policy: a
timeout stops the guard waiting, but it cannot kill a Python thread, and it
bounds nothing about memory, files, child processes or the environment. Code
written by a model needs a process boundary before it can be executed at all.

**Decision:** `sandbox.py` runs a candidate `solution.py` against trusted
acceptance tests in a fresh interpreter, in a temporary directory:

- A stdin handshake: the child starts, waits for one line, and only then
  imports anything. The launcher attaches the OS limits during that window, so
  no candidate code runs before the limits are in force.
- Windows: a Job Object with per-process user-time, process-memory,
  job-memory, active-process and kill-on-close limits. POSIX: `RLIMIT_CPU`,
  `RLIMIT_FSIZE`, a new session, process-group kill, and an RSS watchdog.
- A scrubbed environment rebuilt from an allowlist, with secret-looking names
  dropped and HOME/TEMP/USERPROFILE pointing into the sandbox.
- A Python audit hook inside the child that refuses writes outside the
  sandbox root, refuses network access unless explicitly allowed, and refuses
  child processes.
- A per-run nonce sent over stdin: only a report line carrying it is accepted.
  A missing or forged report is `killed`, never `passed`.

`coding_task.py` supplies the one task (a pure `slugify` function) and the
independent acceptance tests that define success.

**Rationale:**
- **Process boundary first:** memory, CPU and process limits are OS facts, not
  Python promises. The sample cannot talk its way out of them.
- **Least capability for the task:** a pure function needs no network, no
  child process and no host file access, so denying them costs a correct
  solution nothing.
- **Fail closed:** if the job cannot be attached, the run is aborted. If the
  child dies without a nonce-tagged report — including `os._exit(0)` — the
  result is `killed`, not success.
- **Honest scope:** the audit-hook controls are cooperative and the docs say
  so. Result forgery by an in-process sample is acknowledged and left to K02.

**Rejected alternatives:**
- **`subprocess` with a timeout:** the problem statement. It stops one
  process and enforces nothing else.
- **In-process `exec` with patched builtins:** reversible by the sample, and a
  crash takes the host with it.
- **`RLIMIT_AS` for memory:** counts virtual address space; a limit low enough
  to matter breaks interpreter startup on 64-bit Linux. The RSS watchdog is
  sampled enforcement instead, with that limitation stated.
- **`CREATE_SUSPENDED` on Windows:** would require reimplementing the whole
  spawn path to keep the pipes and environment; the stdin handshake gives the
  same ordering with `Popen`.
- **Docker/nsjail:** stronger, but not available hermetically on the target
  machines; `run_python` leaves room for a future backend.

**Trade-offs and boundaries:**
- Audit hooks are not a security boundary against `ctypes` or native code.
- Reads are unrestricted; only writes are checked.
- POSIX memory enforcement is sampled every 50 ms from `/proc` (so Linux
  only), and process count is not capped there (children are killed with the
  group).
- A sample that introspects the harness can read the nonce; result integrity
  against adversarial code is K02's problem, not this task's.
- The interpreter runs with `-I -S -B`: stdlib only, no site-packages, no
  bytecode writes.

## ADR-14: Streaming HTTP API with a Dedicated Loop per Turn

**Status:** Accepted

**Context:** The agent was usable only in-process. An API adds three
requirements the CLI never exercised: token streaming, cancellation that
stops work, and sessions bound to an authenticated user. The obvious
implementation — stream the graph inside the request coroutine and cancel
that task on disconnect — failed in a way that looked correct from the
outside: the socket closed and the unanswered question was rolled back, but
the model kept producing tokens. The cause is in the ASGI server, not in
LangGraph: Starlette runs a response inside an AnyIO cancel scope that,
once cancelled by a disconnect, re-cancels every task in the request tree on
each event-loop iteration. LangGraph's shutdown awaits are interrupted again
as soon as they resume, so the node tasks running the model are never
cancelled.

**Decision:** `api/server.py` exposes one POST per turn as Server-Sent
Events; `api/streaming.py` implements the turn. The turn runs on a fresh
event loop in a worker thread (`ThreadedTurn`). The request coroutine only
forwards events, bridged with `loop.call_soon_threadsafe`, and asks the
worker to cancel when it sees a disconnect or a terminal event. Inside the
worker, `stream_turn` consumes `agent.astream` in an internal task so
`CancelledError` is raised where the graph is iterated; it closes the stream
explicitly and rolls back the unanswered `HumanMessage` after the run has
stopped. A `None` sentinel is published only when all of that has finished.

Sessions are A02 durable threads: the client receives an opaque storage id,
every lookup is scoped by the authenticated user, and another user's id is
`404`. A second concurrent turn on one session is `409`. `_ScopedCheckpointer`
gained async methods that delegate to the sync SQLite saver with
`asyncio.to_thread`, because `SqliteSaver` has no async implementation at
this version.

**Rationale:**
- **Cancellation must reach the work.** A closed response is observability,
  not control. The test double records that its stream was cancelled, and
  the demo records that the token count is frozen after the client leaves.
- **The turn owns its cancellation.** On its own loop it receives exactly
  one `CancelledError`, from an explicit `task.cancel()`, and its cleanup
  cannot be interrupted by the HTTP layer.
- **One loop per turn, not one thread per token.** The worker is idle
  between chunks; the cost is one thread per active turn, and concurrent
  turns do not serialise.
- **Ownership at the storage layer.** Reusing A02 means a mistaken API check
  cannot expose a conversation: the checkpointer refuses a config outside
  the registered owner.
- **Fail closed on configuration.** An empty token map refuses
  construction; there is no anonymous mode.

**Rejected alternatives:**
- **Cancel the response task in place.** Measured to close the socket while
  the model continued; the AnyIO re-cancellation breaks LangGraph's
  shutdown.
- **WebSockets.** No client-to-server frames are needed mid-turn, and SSE
  keeps one request per turn, which makes ownership and the 409 rule
  straightforward.
- **Polling with a server-side buffer.** Adds latency and a second lifetime
  to manage.
- **A process per turn.** Stronger isolation, much larger operational cost;
  the loop-per-turn keeps one process and one store.
- **JWT/OIDC now.** The static token map is enough to prove that an
  authenticated user id reaches the session layer. The authenticator is the
  seam where a real verifier goes; it is not implemented here.

**Trade-offs and boundaries:**
- Text only.
- Static tokens in the environment; no rotation, revocation, rate limiting
  or tenancy beyond a user id.
- `active_turns` is process-local: two replicas could interleave turns for
  one session. One process is the supported deployment.
- The deadline stops the turn, not a tool already running in a thread.
- Latency figures measured with a scripted model over loopback describe the
  API layer, not a provider or a wide-area network.
- A slow client lets the queue between worker and response grow; there is no
  backpressure policy yet.

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

coding_task.py
  └── sandbox.py         (stdlib only; spawns an isolated interpreter)
persistence.py
  ├── langgraph.checkpoint.sqlite  (SqliteSaver)
  ├── bounds.py                    (ToolLimits for guard())
  ├── guard.py                     (ToolGuard)
  └── idempotency.py               (SqliteIdempotencyStore)

api/__main__.py
  └── api/server.py
      ├── api/auth.py              (bearer token → user id)
      ├── api/sessions.py          (user-scoped session service)
      │   └── persistence.py       (DurableMemory, ThreadRecord)
      ├── api/streaming.py         (stream_turn, ThreadedTurn, SSE)
      └── agent/graph.py           (BoundedAgent per turn)
```

Key property: **arrows point in the direction of dependency.** Tools never import agents. Agents never import specific tools. Configuration is imported by everything but imports nothing (except pydantic-settings).

---

## ADR-15: MCP as a Process Boundary, Not an In-Process Function Call

**Status:** Accepted

**Note on numbering:** A05 (in review) introduces ADR-14, so this branch
uses the next number. If A05 is dropped, renumbering this to ADR-14 is a
cosmetic change.

**Context:** The tool library is in-process: `ToolGuard` wraps a Python
function and executes it. That works for functions the application owns, but
it proves nothing about the Model Context Protocol, and it gives the tool the
agent process's full access. The requirement is an MCP server: a separate
program that speaks JSON-RPC over stdio, advertises its tools with
`tools/list`, and executes them with `tools/call`. An HTTP endpoint with
MCP-shaped JSON is not that.

**Decision:** `mcp_tools/server.py` is an MCP server built on the official
Python SDK (`mcp==2.3.0`), launched as a subprocess over stdio at a pinned
protocol revision (`2025-11-25`, forced through the `initialize` handshake).
`mcp_tools/client.py` discovers its tools with `tools/list`, converts each
`inputSchema` into a Pydantic model, and exposes it to the agent as a
LangChain `BaseTool` whose implementation is `tools/call`:

- Both tools are read-only and confined to one `--root` directory. Absolute
  paths, `..` components and symlinks that resolve outside the root are
  refused; files above 1 MB and binary files are skipped or refused.
- A deliberate refusal raises the SDK's `ToolError`, which the protocol
  carries as `isError: true` with the message intact. The bridge raises
  `MCPToolError`; `ToolGuard` turns it into a `ToolMessage` with
  `status="error"`. The model sees the failure; the run does not end.
- The bridge refuses any negotiated revision other than the one it pins, and
  refuses an `inputSchema` it cannot convert to a validating model. Both are
  discovery-time failures, not degraded execution.
- The agent path gives each tool call its own short-lived session, because
  `ToolGuard` executes calls on worker threads with timeouts and an MCP
  session belongs to one event loop. The shared session is used for direct
  async calls.

**Rationale:**
- **The protocol is the demonstration.** Framing, handshake, capability
  negotiation and tool discovery are exercised by tests that write raw
  JSON-RPC to the subprocess's stdin as well as by the SDK client.
- **A separate process is the honest boundary.** The server's root is the
  only tree it will read from, and a crash in the server is a failed call,
  not a failed agent.
- **Fail closed.** An unknown revision, an unrepresentable schema or a
  refused path stops the operation; nothing is silently substituted.
- **Reuse, not a second policy.** The MCP tool is a normal `BaseTool`, so
  the guard's validation, allowlist, budgets and audit trail apply unchanged.

**Rejected alternatives:**
- **Calling the tool functions directly from the agent.** No protocol, and no
  boundary; this is the thing the task exists to replace.
- **FastAPI/HTTP endpoint returning JSON.** Not MCP. Streamable HTTP is a
  real MCP transport, but it would demonstrate HTTP plumbing rather than the
  protocol, and the brief explicitly warns against presenting an HTTP
  endpoint as MCP.
- **A dict `inputSchema` passed straight to LangChain.** `ToolGuard` refuses
  tools without a Pydantic model, and a dict would bypass validation. The
  conversion is the fail-closed choice.
- **One shared MCP session for the whole agent run.** It cannot be used from
  the guard's worker threads without moving the session's event loop into
  another thread. Per-call sessions cost a process spawn and keep the
  threading model clear.

**Trade-offs and boundaries:**
- Per-call spawns are visible in the demo's timings. This is a portfolio
  server, not a pooled production deployment.
- The root is a path check, not a sandbox; a TOCTOU symlink swap is possible
  and the documents say so. Process-level isolation is K01's territory.
- Tools only: no resources, prompts, sampling, elicitation or notifications.
- stdio only: no remote transport, authentication or multi-tenancy.
