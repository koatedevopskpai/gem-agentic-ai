"""One bounded coding task for the isolated runner.

The task is a pure function with a short, fully stated contract and a set of
acceptance tests that are written independently of any candidate solution.
That makes it a safe sample for exercising the sandbox: nothing in the task
needs the network, the host filesystem, child processes or a clock.
"""

from __future__ import annotations

from .sandbox import SandboxLimits, SandboxResult, run_python

__all__ = ["ACCEPTANCE_TESTS", "TASK_ID", "TASK_SPEC", "run_submission"]

TASK_ID = "slugify"

TASK_SPEC = '''\
Implement a pure function in solution.py:

    def slugify(text: str) -> str

Rules:
1. Lowercase the input.
2. Keep ASCII letters (a-z) and ASCII digits (0-9) only.
3. Replace every maximal run of other characters with a single "-".
4. Remove leading and trailing "-".
5. The empty string and strings with no letters or digits produce "".

Examples:
    slugify("Hello, World!")      == "hello-world"
    slugify("  Café  au lait  ")  == "caf-au-lait"
    slugify("A_B--C")             == "a-b-c"
    slugify("Task 42")            == "task-42"
    slugify("---")                == ""
'''

# Written from the contract above, without reference to any candidate.
ACCEPTANCE_TESTS = '''\
from solution import slugify


def test_lowercases_and_joins_words():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_and_whitespace_collapse_to_one_dash():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  a   b  ") == "a-b"


def test_empty_and_punctuation_only_become_empty_string():
    assert slugify("") == ""
    assert slugify("---") == ""
    assert slugify(" !? ") == ""


def test_non_ascii_letters_are_dropped():
    assert slugify("Café au lait") == "caf-au-lait"


def test_underscores_and_hyphens_collapse():
    assert slugify("A_B--C") == "a-b-c"


def test_digits_are_kept():
    assert slugify("Task 42") == "task-42"
    assert slugify("v2.0") == "v2-0"


def test_no_leading_or_trailing_dash():
    result = slugify("--Hello--")
    assert result == "hello"


def test_idempotent():
    for text in ("Hello, World!", "  Café au lait ", "A_B--C", "---"):
        once = slugify(text)
        assert slugify(once) == once
'''


def run_submission(
    source: str,
    *,
    limits: SandboxLimits | None = None,
) -> SandboxResult:
    """Run one candidate ``solution.py`` against the task's acceptance tests."""
    return run_python(source, ACCEPTANCE_TESTS, limits=limits)
