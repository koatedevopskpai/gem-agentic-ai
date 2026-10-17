"""K01: the bounded task's contract and its acceptance tests agree.

These tests exercise the task definition itself. They run candidate code only
through the sandbox, and they check that the written specification, the
acceptance tests and an independent reference implementation agree.
"""

from __future__ import annotations

import ast
import re

import pytest

from gem_agentic_ai.coding_task import ACCEPTANCE_TESTS, TASK_ID, TASK_SPEC, run_submission


def reference_slugify(text: str) -> str:
    """A third implementation, written without ``re``, for cross-checking."""
    result: list[str] = []
    pending_dash = False
    for char in text.lower():
        if ("a" <= char <= "z") or ("0" <= char <= "9"):
            if pending_dash and result:
                result.append("-")
            pending_dash = False
            result.append(char)
        else:
            pending_dash = True
    return "".join(result)


MANUAL_CORRECT = '''\
def slugify(text):
    result = []
    pending_dash = False
    for char in text.lower():
        if ("a" <= char <= "z") or ("0" <= char <= "9"):
            if pending_dash and result:
                result.append("-")
            pending_dash = False
            result.append(char)
        else:
            pending_dash = True
    return "".join(result)
'''


def test_task_has_a_stable_id_and_names_the_entry_point():
    assert TASK_ID == "slugify"
    assert "def slugify(text: str) -> str" in TASK_SPEC
    assert "test_" in ACCEPTANCE_TESTS


def test_an_independent_correct_implementation_passes():
    result = run_submission(MANUAL_CORRECT)
    assert result.status == "passed"
    assert len(result.tests) == 8


def test_every_documented_example_matches_the_reference():
    pattern = re.compile(r"^slugify\((.*?)\)\s*==\s*(.*)$")
    checked = 0
    for line in TASK_SPEC.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        argument = ast.literal_eval(match.group(1))
        expected = ast.literal_eval(match.group(2))
        assert reference_slugify(argument) == expected, argument
        checked += 1
    assert checked == 5


@pytest.mark.parametrize(
    "source,failing_test",
    [
        (
            'import re\n\n\ndef slugify(text):\n'
            '    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")\n',
            "test_lowercases_and_joins_words",
        ),
        (
            'import re\n\n\ndef slugify(text):\n'
            '    return re.sub(r"[^a-z0-9]", "-", text.lower()).strip("-")\n',
            "test_punctuation_and_whitespace_collapse_to_one_dash",
        ),
        (
            'import re\n\n\ndef slugify(text):\n'
            '    return re.sub(r"[^a-z0-9]+", "-", text.lower())\n',
            "test_no_leading_or_trailing_dash",
        ),
        (
            'import re\n\n\ndef slugify(text):\n'
            '    return re.sub(r"\\W+", "-", text.lower()).strip("-")\n',
            "test_non_ascii_letters_are_dropped",
        ),
    ],
    ids=["no-lowercase", "no-collapse", "no-strip", "unicode-kept"],
)
def test_each_contract_rule_is_enforced(source, failing_test):
    result = run_submission(source)
    assert result.status == "failed"
    assert failing_test in [test.name for test in result.tests if not test.passed]
