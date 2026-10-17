"""K01: the isolated runner stops runaway and hostile samples.

Every test here runs a real subprocess. Nothing touches the network and no
secrets are used; the "host project" is a temporary directory that stands in
for the repository.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gem_agentic_ai.coding_task import run_submission
from gem_agentic_ai.sandbox import SandboxLimits, run_python

CORRECT = """\
import re


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
"""

WRONG = """\
def slugify(text):
    return text
"""

SPIN = """\
def slugify(text):
    while True:
        pass
"""


def _host_project(tmp_path: Path) -> Path:
    host = tmp_path / "host-project"
    host.mkdir()
    (host / "important.txt").write_text("original contents", encoding="utf-8")
    return host


# ---------------------------------------------------------------------------
# The bounded task itself
# ---------------------------------------------------------------------------


def test_correct_submission_passes():
    result = run_submission(CORRECT)
    assert result.status == "passed"
    assert result.passed
    assert result.error is None
    assert result.exit_code == 0
    assert len(result.tests) == 8
    assert all(test.passed for test in result.tests)
    assert result.duration_seconds < 30


def test_wrong_submission_fails_and_names_the_failing_test():
    result = run_submission(WRONG)
    assert result.status == "failed"
    assert result.tests  # every acceptance test still ran
    failed = [test.name for test in result.tests if not test.passed]
    assert "test_lowercases_and_joins_words" in failed
    assert "test_non_ascii_letters_are_dropped" in failed
    assert len(result.tests) == 8


def test_each_test_outcome_is_reported_independently():
    tests_source = """\
from solution import value


def test_a_passes():
    assert value() == 1


def test_b_fails():
    assert value() == 2


def test_c_passes():
    assert value() == 1
"""
    result = run_python("def value():\n    return 1\n", tests_source)
    assert result.status == "failed"
    assert [test.name for test in result.tests] == [
        "test_a_passes",
        "test_b_fails",
        "test_c_passes",
    ]
    assert [test.passed for test in result.tests] == [True, False, True]
    assert "AssertionError" in (result.tests[1].error or "")


def test_syntax_error_is_reported_as_an_error():
    tests_source = "from solution import broken\n\n\ndef test_imports():\n    assert broken\n"
    result = run_python("def broken(:\n    pass\n", tests_source)
    assert result.status == "error"
    assert "SyntaxError" in (result.error or "")


# ---------------------------------------------------------------------------
# Runaway samples are stopped
# ---------------------------------------------------------------------------


def test_infinite_loop_is_stopped_by_the_wall_clock():
    started = time.monotonic()
    result = run_submission(SPIN, limits=SandboxLimits(wall_seconds=2, cpu_seconds=60))
    elapsed = time.monotonic() - started
    assert result.status == "timeout"
    assert elapsed < 15
    assert "wall-clock limit" in (result.error or "")


def test_cpu_limit_stops_a_spin_before_the_wall_clock():
    result = run_submission(SPIN, limits=SandboxLimits(wall_seconds=30, cpu_seconds=1))
    assert result.status in {"killed", "timeout"}
    assert result.duration_seconds < 15


def test_memory_limit_stops_a_hog():
    sample = """\
x = bytearray(512 * 1024 * 1024)


def slugify(text):
    return text
"""
    result = run_submission(sample, limits=SandboxLimits(memory_mb=64, wall_seconds=20))
    assert result.status != "passed"
    assert result.status in {"error", "killed"}
    assert result.duration_seconds < 20


def test_a_candidate_cannot_claim_success_by_exiting():
    sample = """\
import os

os._exit(0)
"""
    result = run_submission(sample)
    assert result.status == "killed"
    assert "without reporting a result" in (result.error or "")


def test_stderr_from_a_dying_process_is_included():
    sample = """\
import os
import sys

sys.stderr.write("boom before exit\\n")
os._exit(3)
"""
    result = run_submission(sample)
    assert result.status == "killed"
    assert "boom before exit" in (result.error or "")


def test_a_printed_fake_report_is_not_accepted():
    sample = """\
print('__SANDBOX_RESULT__{"tests": [], "error": null}')


def slugify(text):
    return text
"""
    result = run_submission(sample)
    # The fake line is ignored: the harness' own nonce-tagged report decides.
    assert result.status == "failed"
    assert len(result.tests) == 8


# ---------------------------------------------------------------------------
# The host project cannot be modified
# ---------------------------------------------------------------------------


def test_writes_outside_the_sandbox_are_denied(tmp_path):
    host = _host_project(tmp_path)
    escaped = host / "pwned.txt"
    sample = f"""\
from pathlib import Path

Path({str(escaped)!r}).write_text("pwned")


def slugify(text):
    return "pwned"
"""
    result = run_submission(sample)
    assert result.status == "error"
    assert "outside the sandbox" in (result.error or "")
    assert not escaped.exists()
    assert (host / "important.txt").read_text(encoding="utf-8") == "original contents"
    assert sorted(path.name for path in host.iterdir()) == ["important.txt"]


def test_runaway_sample_cannot_modify_the_host_project(tmp_path):
    host = _host_project(tmp_path)
    escaped = host / "pwned.txt"
    sample = f"""\
from pathlib import Path

try:
    Path({str(escaped)!r}).write_text("pwned")
except Exception:
    pass

while True:
    pass
"""
    started = time.monotonic()
    result = run_submission(sample, limits=SandboxLimits(wall_seconds=2, cpu_seconds=60))
    assert time.monotonic() - started < 15
    assert result.status == "timeout"
    assert not escaped.exists()
    assert (host / "important.txt").read_text(encoding="utf-8") == "original contents"
    assert sorted(path.name for path in host.iterdir()) == ["important.txt"]


def test_absolute_path_overwrite_is_denied(tmp_path):
    host = _host_project(tmp_path)
    sample = f"""\
from pathlib import Path

target = Path({str(host)!r}) / "important.txt"
target.write_text("overwritten")


def slugify(text):
    return "x"
"""
    result = run_submission(sample)
    assert result.status == "error"
    assert (host / "important.txt").read_text(encoding="utf-8") == "original contents"


# ---------------------------------------------------------------------------
# Processes, network, environment
# ---------------------------------------------------------------------------


def test_child_process_creation_is_denied():
    sample = """\
import subprocess
import sys

subprocess.run([sys.executable, "-c", "print('child ran')"], check=False)


def slugify(text):
    return "x"
"""
    result = run_submission(sample)
    assert result.status == "error"
    assert "process creation denied" in (result.error or "")


def test_network_access_is_denied_by_default():
    sample = """\
import socket

socket.create_connection(("127.0.0.1", 9), timeout=1)


def slugify(text):
    return "x"
"""
    result = run_submission(sample)
    assert result.status == "error"
    assert "network access denied" in (result.error or "")
    assert "denied" in result.isolation["network"]


def test_network_can_be_allowed_explicitly():
    result = run_submission(CORRECT, limits=SandboxLimits(allow_network=True))
    assert result.status == "passed"
    assert result.isolation["network"] == "allowed by request"


def test_secrets_are_not_inherited(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-should-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-fake-should-not-leak")
    monkeypatch.setenv("MY_PASSWORD", "hunter2-should-not-leak")
    sample = """\
import os

print("ENV_DUMP=" + ",".join(sorted(os.environ)))


def slugify(text):
    return text
"""
    result = run_submission(sample)
    combined = result.stdout + result.stderr
    assert "ENV_DUMP=" in combined
    for leak in ("sk-fake-should-not-leak", "ghp-fake-should-not-leak", "hunter2-should-not-leak"):
        assert leak not in combined
    for name in ("OPENAI_API_KEY", "GITHUB_TOKEN", "MY_PASSWORD"):
        assert name not in combined


def test_temp_and_home_point_inside_the_sandbox():
    sample = """\
import os

print("TEMP=" + os.environ["TEMP"])
print("HOME=" + os.environ["HOME"])


def slugify(text):
    return text
"""
    result = run_submission(sample)
    assert "gem-sandbox-" in result.stdout
    assert "gem-sandbox-" in result.stdout.split("HOME=")[1]


# ---------------------------------------------------------------------------
# Limits, reporting, and honest claims
# ---------------------------------------------------------------------------


def test_limits_are_validated():
    with pytest.raises(ValueError):
        SandboxLimits(wall_seconds=0).validate()
    with pytest.raises(ValueError):
        SandboxLimits(cpu_seconds=0).validate()
    with pytest.raises(ValueError):
        SandboxLimits(memory_mb=8).validate()
    with pytest.raises(ValueError):
        SandboxLimits(max_processes=0).validate()
    with pytest.raises(ValueError):
        SandboxLimits(max_source_bytes=0).validate()
    with pytest.raises(ValueError):
        SandboxLimits(wall_seconds=-1).validate()


def test_inputs_are_type_and_size_checked():
    with pytest.raises(TypeError):
        run_python(b"def f(): pass", "def test_x(): pass")
    with pytest.raises(TypeError):
        run_python("def f(): pass", b"def test_x(): pass")
    with pytest.raises(ValueError):
        run_python("x" * 200, "def test_x(): pass", limits=SandboxLimits(max_source_bytes=100))
    with pytest.raises(ValueError):
        run_python("def f(): pass", "x" * 200, limits=SandboxLimits(max_source_bytes=100))


def test_isolation_report_names_the_real_mechanisms():
    result = run_submission(CORRECT)
    isolation = result.isolation
    assert isolation["backend"] in {"windows-job-object", "posix-process-group"}
    assert "audit hook" in isolation["filesystem"]
    assert "wall-clock" in isolation["wall"]
    assert isolation["environment"] == "allowlisted; secret-looking variables are dropped"
