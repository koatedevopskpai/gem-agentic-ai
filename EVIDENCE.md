# Evidence index

A reviewer's route through this repository. Everything here points at code,
tests, or a dated run recorded in [`evidence/`](evidence/). No claim is
asserted without a location.

**Five-minute path:** this file → [What it is](#what-it-is) →
[strongest measured results](#the-strongest-measured-results) →
[how to reproduce](#how-to-reproduce) →
[what this does not establish](#what-this-does-not-establish).

---

## What it is

An agent-orchestration reference implementation with two ReAct paths — a
manual thought-action-observation loop and a LangGraph state machine — behind
one shared `ToolGuard`. It adds durable, user-scoped conversation memory and a
bounded, isolated runner for one small coding task. The offline suite uses a
scripted model; CI runs lint and tests without a provider key.

| | |
|---|---|
| Scope | **Public reference build.** Self-funded; no customer deployment, no hosted service |
| Author | Koate Kpai — https://github.com/koatedevopskpai |
| Entry point | `src/gem_agentic_ai/`, `python -m gem_agentic_ai demo` |
| Revision described | `52e9398` (merge of C03, A01, A02, A03 and K01) |
| Evidence date | 17 October 2026 |

**Scope boundary.** This index describes the merged tree at `52e9398` and the
tasks in it. Branches that are not merged are not claimed here. The two agent
implementations, the tool library, configuration and CLI pre-date this work;
the contribution below is the execution policy, durability, isolation, and
the test/CI layer around them.

## My contribution by module

| Module | What I added or changed | Task | Tests |
|---|---|---|---|
| `guard.py`, `bounds.py`, `idempotency.py`, `tools/actions.py` | One execution policy for both agent paths: allowlist, schema refusal, argument-size and observation-token limits, per-call timeout, call budgets, circuit breaker, retries, idempotency, approval | A03 | `tests/test_tool_guard.py`, `tests/test_bounds.py`, `tests/test_bounded_execution.py` |
| `agent/simple.py`, `agent/graph.py` | Every tool call routed through `GuardedTool`; the run reports its budget and stop reason | A03 | `tests/test_bounded_execution.py`, `tests/test_agent_execution.py` |
| `persistence.py` | `DurableMemory`: SQLite checkpoints, an opaque thread id per `(user_id, thread_key)`, a checkpointer that authorises every read and write, deletion, retention, and a SQLite-backed guard | A02 | `tests/test_persistence.py` (20) |
| `sandbox.py`, `coding_task.py` | A bounded `slugify` task and a runner that enforces wall/CPU/memory limits through the OS, scrubs the environment, denies writes/network/children with audit hooks, and refuses a self-certified result | K01 | `tests/test_sandbox.py` (22), `tests/test_coding_task.py` (7) |
| `.github/workflows/ci.yml`, `tests/fakes.py`, `tests/test_agent_execution.py` | Pinned CI (lint + offline tests), a scripted chat model, and end-to-end agent tests | A01 | `tests/test_agent_execution.py` |
| `tests/conftest.py`, README install steps | The clean-checkout fix: seed the key at import time; document `pip install --no-deps -e .` | C03 | full suite from a clean environment |

Design records: [ADR-11](docs/ARCHITECTURE.md) (one shared guard),
[ADR-12](docs/ARCHITECTURE.md) (durable memory), [ADR-13](docs/ARCHITECTURE.md)
(isolated runner), and the task notes [docs/A02.md](docs/A02.md),
[docs/A03.md](docs/A03.md), [docs/K01.md](docs/K01.md).

---

## The strongest measured results

**Merged tree, 17 October 2026** — full offline suite, from
[`evidence/C04.md`](evidence/C04.md):

| Platform | Result | Coverage |
|---|---|---|
| Linux (`python:3.11-slim`, Python 3.11.17) | **266 passed, 1 deselected** | **89%** |
| Windows 11 (Python 3.11.9) | **266 passed, 1 deselected** | 89% |

The suite grew with A05 (async streaming API) and A06 (MCP tool server), so the
count moved from **177 passed at 93%** — recorded when this pack was written — to
**266 passed at 89%**. Coverage falls as new code lands; both figures are
reported, and neither is gated. The Linux run remains the authoritative one.

Per module on Linux: `persistence.py` 100%, `bounds.py` 98%, `guard.py` 93%,
`sandbox.py` 90%. On Windows `sandbox.py` is 85% because the POSIX-only
branches do not execute there — which is why the Linux run is the
authoritative one. Coverage is reported, not gated; the number is not a
quality claim.

**Isolation controls, measured** — seven hostile samples against a sentinel
host directory ([`evidence/K01.md`](evidence/K01.md) §5). This is the
strongest behavioural result in the repository because each row states what
actually stopped the sample:

| Sample | Result | What stopped it |
|---|---|---|
| Correct submission | `passed` | — |
| Infinite loop | `timeout` after 2.01 s | wall-clock limit, then process-group/job termination |
| Write to the host project | `error` | audit hook; sentinel file absent and unchanged |
| 512 MiB allocation under a 64 MiB limit | `killed` (Linux) / `MemoryError` (Windows) | RSS watchdog / Job Object memory limit |
| Spawn a child process | `error` | audit hook / Windows `ActiveProcessLimit=1` |
| Open a network connection | `error` | audit hook (not an OS boundary) |
| Print a fake success report | `failed` | results are a nonce-tagged report, not process output |

Read this as *controls that fired on one Windows host and one Linux
container*, not as a security proof. The mechanisms and their limits are in
[docs/K01.md](docs/K01.md).

**Durability and replay** — seven A02 acceptance tests and seven A03
acceptance tests pass offline, including restart/resume, cross-user refusal,
deletion, retention, and one logical side effect on replay
([`evidence/A02.md`](evidence/A02.md), [`evidence/A03.md`](evidence/A03.md)).

### A claim I removed

An earlier CV/portfolio line said: *"Added plan fingerprints, an execution
ledger and step budgets to bound repeated work."* The C01 claim register
classified it **UNSUPPORTED** — zero matches in source or README — and it was
removed from the CV wording, along with the skill lines listing "execution
ledgers; plan fingerprints; step budgets". It was not reworded or softened.

This repository never had that feature either. A history-wide search returns
zero commits for each phrase:

```bash
git log --all --oneline -S"execution ledger"      # 0
git log --all --oneline -S"plan fingerprint"      # 0
git log --all --oneline -S"step budget"           # 0
git grep -niE "execution ledger|plan fingerprint|step budget" HEAD   # 0
```

The word "ledger" does appear in two unrelated places: the in-memory demo
ticket list in `tools/actions.py`, and prose describing the idempotency
records. Neither is an execution ledger. Correcting a claim downward is part
of the evidence trail, not a footnote to it.

---

## How to reproduce

Offline, free, no provider key — this is what CI runs:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pip install --no-deps -e .        # src layout
ruff check .
pytest -m "not integration" -q
```

Targeted groups:

```bash
pytest tests/test_tool_guard.py tests/test_bounded_execution.py -q   # A03
pytest tests/test_persistence.py -q                                  # A02
pytest tests/test_sandbox.py tests/test_coding_task.py -q            # K01
```

The Linux figures come from the same commands in a `python:3.11-slim`
container with the worktree mounted at `/work`.

**Live, billable, not part of this evidence.** `python -m gem_agentic_ai demo`
(or `chat`, `simple`) needs `OPENAI_API_KEY` and calls a model provider.
`pytest -m integration` calls DuckDuckGo. Neither was run for this pack, and
the integration test is deselected in CI as well. No deployment, hosted
service or uptime claim exists to reproduce.

---

## One success trace, one failure

**Success — state survives a restart.** The test opens `DurableMemory` over a
directory, records "My name is Ada", closes the store, opens a second
`DurableMemory` over the same directory, and asks "What is my name?". The
second process resolves the same opaque storage id, and the message list the
model receives on its first turn is:

```text
My name is Ada.
Noted, Ada.
What is my name?
```

Command and result:

```text
$ pytest tests/test_persistence.py::TestRestartAndResume::test_conversation_survives_a_process_restart -v --no-cov
tests/test_persistence.py::...::test_conversation_survives_a_process_restart PASSED
============================== 1 passed in 0.63s ==============================
```

**Failure — the POSIX path failed the first CI run, and the fix is recorded.**
K01's first Linux CI run failed 25 tests with
`TypeError: 'Event' object is not callable`. Root cause: the watchdog stored
its flag as `self._stop`, shadowing `threading.Thread._stop()`, which CPython
calls while joining; the watchdog exists only on POSIX, so the Windows-only
run could never surface it. Renaming the attribute fixed it (`a457ae7`); no
test was skipped or weakened. Trace, root cause and re-run:
[`evidence/K01.md`](evidence/K01.md) §2b. This is why the Linux run, not the
Windows run, is quoted as authoritative above.

**A failure the design blocks.** Invalid arguments, a tool with no schema, an
unknown tool, and a call over budget are refused before execution and recorded
with their status; the model sees an error, not a silent success
(`tests/test_tool_guard.py`, `tests/test_bounded_execution.py`).

---

## Decisions and trade-offs

| Decision | Where |
|---|---|
| One shared `ToolGuard` instead of per-path middleware; refusal semantics and stop reasons | [docs/A03.md](docs/A03.md), ADR-11 |
| Durable memory: opaque thread ids, ownership enforced in the checkpointer, explicit retention | [docs/A02.md](docs/A02.md), ADR-12 |
| Isolated runner: OS limits where possible, audit hooks where not; a timeout alone is not isolation | [docs/K01.md](docs/K01.md), ADR-13 |
| Testing: scripted model, offline CI, coverage reported not gated | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ADR-8, `docs/CLEAN_CHECKOUT.md` |
| Clean-checkout baseline and the import-time key fixture | [docs/CLEAN_CHECKOUT.md](docs/CLEAN_CHECKOUT.md) |

---

## What this does not establish

- **Exactly-once tools.** A completed action is recorded by
  `(tool, action key)`. A tool without a stable key is at-least-once across a
  crash; a crash between the external side effect and the idempotency write
  repeats the action (the dual-write window). `SqliteIdempotencyStore` is
  single-host and first-writer-wins, not a distributed lock: two processes
  racing the same key can both execute. Nothing here claims exactly-once
  delivery to an external system.
- **A durable database.** Memory is **one unencrypted SQLite file** set on
  one host (`checkpoints.sqlite`, `thread-registry.sqlite`,
  `idempotency.sqlite`). Anyone with filesystem access can read it. Ownership
  is enforced only for an application-supplied authenticated `user_id`;
  there is no at-rest encryption, no per-user database, and no async
  checkpointer in A02.
- **That a timed-out tool stops.** The per-call timeout bounds how long the
  guard waits, **not the process**: the tool runs in a daemon thread, Python
  cannot interrupt it, and it may continue and perform its side effect late
  after its result is discarded. This is A03's documented limitation, not an
  unobserved edge case.
- **That the guard is a sandbox.** `ToolGuard` decides *which calls may run*;
  it does not restrict what a running tool can read, write, connect to, or
  spawn. Isolation exists only in the separate, narrow K01 runner: OS limits
  are real, but the filesystem/network/process denials are cooperative Python
  audit hooks that `ctypes` or a native extension can bypass; reads are
  unrestricted; POSIX memory is a 50 ms sampled watchdog, not a hard cap;
  POSIX process count is not capped; a sample that introspects the harness can
  read the nonce and forge a report; macOS is untested. It is not
  container-grade isolation.
- **Provider quality, latency or cost.** No live model call is recorded here.
  Model latency and spend are outside the guard entirely.
- **Production anything.** No deployment, uptime, customer, scale or
  multi-tenant claim. Test counts are one host and one Linux container on
  17 October 2026, and coverage is reported, not gated.
