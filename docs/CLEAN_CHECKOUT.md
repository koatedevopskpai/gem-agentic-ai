# Clean-checkout baseline

Recorded 17 October 2026 for task **C03** of the portfolio improvement
backlog. The purpose is to prove that another engineer can install and run
the offline example from a fresh clone, without credentials and without a
local `.env`.

Base commit: `8be104b` (`main` before this task).
This document describes the branch `task/C03-clean-checkout-baseline`.

## Runtime

| Item | Value |
|---|---|
| Required Python | 3.11+ (`requires-python = ">=3.11"`) |
| `.python-version` | 3.11 |
| Baseline measured on | CPython 3.11.9, Windows 11 |
| CI target | CPython 3.11 on Ubuntu (`ubuntu-latest`) |

## Install

From a fresh clone, with no `.env` file and no API key:

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements-dev.txt   # pinned runtime + test tooling
pip install --no-deps -e .            # src layout: makes the package importable
```

`requirements.txt` is the pin set:

```
langchain==1.3.8
langchain-core==1.4.6
langchain-openai==1.3.0
langgraph==1.2.4
duckduckgo_search==8.1.1
pydantic==2.13.4
pydantic-settings==2.14.1
```

`pip install -e ".[dev]"` also works, but resolves the newest versions
matching the ranges in `pyproject.toml` rather than these pins. On
17 October 2026 that produced langchain 1.4.6 / langgraph 1.2.16, i.e. a
different environment from the one described here. CI therefore installs
the pinned set.

## Required offline check

```bash
python -m pytest -m "not integration"
```

- No network, no credentials and no `.env` file are required. `tests/conftest.py`
  sets a placeholder `OPENAI_API_KEY` before collection so that
  `gem_agentic_ai.config` can import; no test makes a live model call.
- The LLM is scripted or mocked in every agent test. Tool calls in the
  simple ReAct loop are dispatched to real tool objects in the tests that
  were added under A01.
- CI (`.github/workflows/ci.yml`) runs this exact command on every push and
  pull request, with no secrets configured. Integration-marked tests are not
  run there.

## Optional checks (not part of the default run)

| Check | Command | Why it is separate |
|---|---|---|
| Live web search | `python -m pytest -m integration` | Calls DuckDuckGo; needs network and is rate-limited |
| Live model run | `python -m gem_agentic_ai demo` | Needs a real `OPENAI_API_KEY` and spends tokens |

The default run excludes both. A skipped or deselected integration test is
not reported as live validation.

## Recorded baseline

Command:

```bash
python -m pytest -m "not integration" -q
```

Result on 17 October 2026, CPython 3.11.9, clean venv, no `.env`:

```
collected 70 items / 1 deselected / 69 selected
69 passed, 1 deselected in 8.40s
TOTAL  232 statements  23 missed  90%
```

Per-file coverage at this commit: every module 100% except
`main.py` at 76% (the interactive `run_chat` / `run_simple_demo` bodies).

## Limitations

- The baseline was measured on Windows; CI runs Ubuntu. The dependency set
  is pure Python at these pins, but the CI run is the authoritative record.
- No live model call was executed, so this baseline says nothing about
  answer quality or live tool use — only that the offline suite and the
  tool-layer behaviour it covers pass.
- Coverage is reported, not enforced. The number is not a quality gate and
  will move as tests change.
