# Gem Agentic AI

[![ci](https://github.com/koatedevopskpai/gem-agentic-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/koatedevopskpai/gem-agentic-ai/actions/workflows/ci.yml)

An LLM agent orchestration framework demonstrating two architectural approaches to the ReAct (Reasoning + Acting) pattern:

1. **Simple ReAct Loop** — a manual thought-action-observation loop showing the core pattern from first principles.
2. **LangGraph State Machine** — a formal state graph with checkpointing, streaming, and typed state management.

Both implementations consume the same shared tool library and the same
`ToolGuard`, illustrating dependency inversion and one execution policy for
both paths. The guard bounds every tool call: schema validation, allowlist,
timeouts, call and token budgets, a per-tool circuit breaker, retry rules,
idempotency and explicit approval for consequential actions. See
[Bounded Execution](#bounded-execution) for what is and is not covered.

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
| Tool execution | Shared `ToolGuard` | One policy for both agent paths: validate, bound, approve, de-duplicate |
| Idempotency | In-memory or SQLite store | A replayed action produces one logical result; SQLite survives a restart |
| Conversation memory | `MemorySaver` or SQLite via `DurableMemory` | The default is in-process; the SQLite checkpointer survives a restart and scopes threads to a user |
| CLI | `argparse` (stdlib) | Zero dependencies; explicit and easy to follow |
| Logging | `logging` module | Structured output with levels; production standard |
| Testing | `pytest` + `pytest-cov` | Offline suite with a scripted model; coverage reported by CI, not gated |

---

## Bounded Execution

A model can ask for a tool call, but it cannot decide whether the call is
allowed. Every call on both agent paths passes through `ToolGuard`, which
applies one policy and records the outcome in an audit trail.

| Bound | Default | When reached |
|-------|---------|--------------|
| Wall-clock deadline for the run | 30 s | call refused, `deadline_exceeded` |
| Per-call timeout | 5 s | error observation, `timeout` |
| Total tool calls | 12 | call refused, `call_budget_exceeded` |
| Calls per tool | 4 | that tool refused, `tool_call_limit` |
| Attempts per call | 2 | retryable errors only; unknown exceptions run once |
| Consecutive failures per tool | 3 | circuit opens, `circuit_open` |
| Accumulated observation size | ~4,000 tokens (estimated) | call refused, `token_budget_exceeded` |
| Serialised argument size | 4,000 bytes | call refused, `arguments_too_large` |

Invalid arguments are refused before the tool runs. A tool without a Pydantic
argument schema is refused rather than executed unvalidated. Consequential
tools require an explicit approval for their action key and are never retried
automatically after a timeout.

```python
from gem_agentic_ai.agent.simple import run_simple_agent
from gem_agentic_ai.bounds import ToolLimits
from gem_agentic_ai.guard import ToolGuard

guard = ToolGuard(
    ToolLimits(
        deadline_seconds=20,
        max_total_calls=6,
        max_calls_per_tool=2,
        consequential_tools=frozenset({"create_support_ticket"}),
    )
)
result = run_simple_agent(llm_with_tools, question, tools_by_name, guard=guard)

print(result.answer)
print(result.budget.stop_reason)  # None, or the first bound that was hit
```

The first refusal sets `result.budget.stop_reason` and is never cleared, so a
model that answers after a refused call cannot hide the fact that a call was
cut off. `result.budget.records` lists every call with its status, attempt
count and duration.

An action can be replayed safely. The guard checks an idempotency store before
executing and records the completed result afterwards, keyed by
`(tool, action key)`. `InMemoryIdempotencyStore` covers one process;
`SqliteIdempotencyStore` survives a restart.

**What this does not establish.** The per-call timeout stops the guard
waiting; it does not kill the tool. Python cannot interrupt a thread, so a
non-cooperative tool may keep running after its result is discarded. Model
latency and provider spend are outside this layer. The guard is an execution
policy, not a security sandbox. The design and its limits are recorded in
[docs/A03.md](docs/A03.md).

---

## Durable Memory

`MemorySaver` forgets a conversation when the process exits. `DurableMemory`
stores LangGraph checkpoints in a local SQLite file and puts a user boundary
around every thread.

```python
from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.persistence import DurableMemory

memory = DurableMemory("var/agent-memory")          # checkpoints + idempotency
config = memory.config("alice", "support-1")        # opaque thread id + owner
agent = build_agent(llm, tools, checkpointer=memory.checkpointer)

agent.invoke({"messages": [("user", "My name is Ada")]}, config)
# ... process restarts ...
agent.invoke({"messages": [("user", "What is my name?")]}, config)
```

| Operation | Behaviour |
|---|---|
| `config(user_id, thread_key)` | Resolves a stable, opaque storage id for that pair; registers the thread on first use |
| Checkpoint read/write | Refused unless the config carries both `user_id` and the registered `thread_id` |
| `list_threads(user_id)` | Only that user's threads, most recently used first |
| `delete_thread(user_id, thread_key)` | Removes checkpoints and the registry row; another user's key returns `False` |
| `sweep_expired()` | Removes threads idle longer than `retention_days` (default 30); opening the store never deletes anything |
| `guard()` | A `ToolGuard` backed by `SqliteIdempotencyStore` in the same directory |

A restart replays from the last completed checkpoint. Completed tool actions
are recorded in the idempotency store, so a step replayed after a crash
produces one logical side effect, not two. Approval decisions are deliberately
**not** durable: they record a human decision and must be made again in a new
process.

**What this does not establish.** This is one SQLite file on one host, not a
distributed store or a lock across machines. Thread ownership is enforced by
the checkpointer using the caller-supplied `user_id`; an application that lets
a caller choose that value must authenticate it first. The async checkpointer
path is not implemented at this version. The design and its limits are
recorded in [docs/A02.md](docs/A02.md).

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
├── bounds.py            # ExecutionBudget, ToolLimits, stop reasons
├── guard.py             # ToolGuard and GuardedTool (validation, retries, approval)
├── idempotency.py       # In-memory and SQLite idempotency stores
├── persistence.py       # Durable, user-scoped SQLite conversation memory
├── config.py            # pydantic-settings configuration
├── main.py              # CLI argument parsing and mode dispatch
├── tools/
│   ├── __init__.py      # Tool registry / re-exports
│   ├── system.py        # Standard-library tools (clock, calculator)
│   ├── web.py           # DuckDuckGo web search
│   ├── database.py      # In-memory SQLite employee database
│   └── actions.py       # Consequential demo action (approval + idempotency)
├── agent/
│   ├── __init__.py      # Agent registry / re-exports
│   ├── simple.py        # Manual ReAct loop
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
| Persistence | None (stateless) | MemorySaver (in-memory) or SQLite via `DurableMemory` |
| Streaming | Blocking (full response) | Event-based streaming |
| Tool routing | Manual `if/else` | Built-in conditional edges |
| Execution bounds | Shared `ToolGuard` | Shared `ToolGuard` |
| Approval gate | Explicit approval for consequential tools | Explicit approval for consequential tools |
| Code complexity | Small, single-file loop (pedagogical) | Graph builder plus guard integration |

---

## License

This project is provided as a portfolio piece. See LICENSE file for details.
