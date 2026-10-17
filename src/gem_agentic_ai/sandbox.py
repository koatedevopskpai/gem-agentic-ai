"""Ephemeral, resource-bounded runner for one small piece of Python code.

The runner executes a candidate ``solution.py`` plus a trusted
``acceptance_tests.py`` in a fresh temporary directory, in a separate Python
interpreter, with:

- a scrubbed environment (allowlist; nothing that looks like a secret),
- OS-enforced wall-clock, CPU, memory and process limits,
- in-process guards that deny writes outside the sandbox and deny network
  access and child processes by default.

**This is not a complete security boundary.** The OS limits are real, but the
filesystem and network restrictions are Python audit hooks: code that reaches
``ctypes`` or a native extension can bypass them. See ``docs/K01.md`` for the
threat model before treating this as a hostile-code sandbox.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SandboxError", "SandboxLimits", "SandboxResult", "TestOutcome", "run_python"]

_RESULT_MARKER = "__SANDBOX_RESULT__"
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "PRIVATE")
_ENV_ALLOWLIST = {
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "OS",
    "LANG",
    "LC_ALL",
    "TZ",
    "TERM",
}


class SandboxError(RuntimeError):
    """Raised when the runner cannot establish its limits (fail closed)."""


@dataclass(frozen=True)
class SandboxLimits:
    """Explicit resource limits for one run."""

    wall_seconds: float = 5.0
    cpu_seconds: int = 3
    memory_mb: int = 256
    max_processes: int = 1
    allow_network: bool = False
    max_source_bytes: int = 100_000

    def validate(self) -> None:
        if not isinstance(self.wall_seconds, (int, float)) or self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be a positive number")
        if not isinstance(self.cpu_seconds, int) or self.cpu_seconds < 1:
            raise ValueError("cpu_seconds must be a positive int")
        if not isinstance(self.memory_mb, int) or self.memory_mb < 16:
            raise ValueError("memory_mb must be an int >= 16")
        if not isinstance(self.max_processes, int) or self.max_processes < 1:
            raise ValueError("max_processes must be an int >= 1")
        if not isinstance(self.max_source_bytes, int) or self.max_source_bytes < 1:
            raise ValueError("max_source_bytes must be a positive int")


@dataclass
class TestOutcome:
    name: str
    passed: bool
    error: str | None = None


@dataclass
class SandboxResult:
    """Outcome of one run. ``status`` is one of passed/failed/timeout/killed/error."""

    status: str
    exit_code: int | None
    duration_seconds: float
    stdout: str
    stderr: str
    tests: list[TestOutcome] = field(default_factory=list)
    error: str | None = None
    isolation: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


_HARNESS = r'''"""Sandbox harness: install guards, run acceptance tests, report JSON."""
import json
import os
import sys

MARKER = "__SANDBOX_RESULT__"
ROOT = os.path.realpath(os.getcwd())
ALLOW_NETWORK = sys.argv[1] == "1"
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
PATH_EVENTS = {
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.mkdir",
    "os.chmod",
    "os.chown",
    "os.truncate",
    "os.link",
    "os.symlink",
    "os.utime",
}
PROCESS_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn"}
NETWORK_EVENTS = {
    "socket.connect",
    "socket.bind",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
}


def _within_root(path):
    try:
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, (str, os.PathLike)):
            return True
        target = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    try:
        return os.path.commonpath((ROOT, target)) == ROOT
    except ValueError:
        return False


def _check_write(path):
    if not _within_root(path):
        raise PermissionError("write outside the sandbox is denied: %r" % (path,))


def _hook(event, args):
    if event == "open":
        path, mode, flags = args
        writing = bool(mode and any(char in mode for char in "wax+"))
        writing = writing or bool(flags and flags & WRITE_FLAGS)
        if writing:
            _check_write(path)
    elif event in PATH_EVENTS:
        for value in args:
            if isinstance(value, (str, bytes, os.PathLike)):
                _check_write(value)
    elif event in PROCESS_EVENTS:
        raise PermissionError("process creation denied by the sandbox")
    elif event in NETWORK_EVENTS and not ALLOW_NETWORK:
        raise PermissionError("network access denied by the sandbox")


def main():
    # Do not import or run anything until the launcher has attached the OS
    # limits to this process. The nonce is delivered here rather than in a file
    # so that a candidate cannot copy it out of the sandbox.
    nonce = sys.stdin.readline().split()[1]
    # -I keeps the working directory and user site-packages off sys.path; the
    # sandbox files themselves are the one directory we add back.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.addaudithook(_hook)
    tests = []
    error = None
    try:
        import acceptance_tests
    except BaseException as exc:  # report any import-time failure, including SystemExit
        error = "%s: %s" % (type(exc).__name__, exc)
    else:
        names = sorted(
            name
            for name, value in vars(acceptance_tests).items()
            if name.startswith("test_") and callable(value)
        )
        for name in names:
            try:
                getattr(acceptance_tests, name)()
            except BaseException as exc:  # a failed test is data, not a crash
                tests.append(
                    {"name": name, "passed": False, "error": "%s: %s" % (type(exc).__name__, exc)}
                )
            else:
                tests.append({"name": name, "passed": True, "error": None})
    print(MARKER + nonce + " " + json.dumps({"tests": tests, "error": error}))


main()
'''


def _build_env(root: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper in _ENV_ALLOWLIST and not any(marker in upper for marker in _SECRET_MARKERS):
            env[name] = value
    env.update(
        {
            "TEMP": root,
            "TMP": root,
            "TMPDIR": root,
            "HOME": root,
            "USERPROFILE": root,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
        }
    )
    return env


def _isolation_report(limits: SandboxLimits) -> dict[str, str]:
    windows = sys.platform == "win32"
    return {
        "backend": "windows-job-object" if windows else "posix-process-group",
        "filesystem": (
            "writes restricted to the temporary directory by a Python audit hook; "
            "reads are not restricted"
        ),
        "network": (
            "allowed by request"
            if limits.allow_network
            else "denied by a Python audit hook (not an OS-level boundary)"
        ),
        "processes": (
            f"Windows Job Object ActiveProcessLimit={limits.max_processes}"
            if windows
            else "process group killed on exit; RLIMIT_NPROC is not enforced"
        ),
        "memory": (
            f"Windows Job Object ProcessMemoryLimit={limits.memory_mb} MiB"
            if windows
            else f"RSS watchdog at {limits.memory_mb} MiB (50 ms sampling)"
        ),
        "cpu": (
            f"Windows Job Object PerProcessUserTimeLimit={limits.cpu_seconds}s"
            if windows
            else f"RLIMIT_CPU={limits.cpu_seconds}s"
        ),
        "wall": f"{limits.wall_seconds}s wall-clock kill",
        "environment": "allowlisted; secret-looking variables are dropped",
    }


# ---------------------------------------------------------------------------
# Windows Job Object limits
# ---------------------------------------------------------------------------

if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
    _JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL

    class _WindowsJob:
        """Owns a job-object handle; terminate and close are idempotent."""

        def __init__(self, handle: int):
            self.handle = handle

        def __bool__(self) -> bool:
            return bool(self.handle)

        def terminate(self) -> None:
            if self.handle:
                _kernel32.TerminateJobObject(self.handle, 1)

        def close(self) -> None:
            if self.handle:
                _kernel32.CloseHandle(self.handle)
                self.handle = 0

    def _create_job(limits: SandboxLimits) -> _WindowsJob:
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            error = ctypes.get_last_error()
            raise SandboxError(f"CreateJobObject failed: {ctypes.WinError(error)}")
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_PROCESS_TIME
            | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | _JOB_OBJECT_LIMIT_JOB_MEMORY
            | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info.BasicLimitInformation.PerProcessUserTimeLimit = limits.cpu_seconds * 10_000_000
        info.BasicLimitInformation.ActiveProcessLimit = limits.max_processes
        info.ProcessMemoryLimit = limits.memory_mb * 1024 * 1024
        info.JobMemoryLimit = limits.memory_mb * 1024 * 1024
        if not _kernel32.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            error = ctypes.get_last_error()
            _kernel32.CloseHandle(job)
            raise SandboxError(f"SetInformationJobObject failed: {ctypes.WinError(error)}")
        return _WindowsJob(int(job))

    def _assign_job(job: _WindowsJob, proc: subprocess.Popen) -> None:
        handle = getattr(proc, "_handle", None)
        if not handle:
            raise SandboxError("cannot obtain the child process handle")
        if not _kernel32.AssignProcessToJobObject(job.handle, int(handle)):
            raise SandboxError(
                f"AssignProcessToJobObject failed: {ctypes.WinError(ctypes.get_last_error())}"
            )


# ---------------------------------------------------------------------------
# POSIX limits: rlimits plus an RSS watchdog
# ---------------------------------------------------------------------------

if os.name == "posix":  # pragma: no cover - exercised on POSIX only
    import resource

    def _posix_preexec(limits: SandboxLimits):
        def apply() -> None:
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            file_limit = limits.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))

        return apply

    def _group_rss_mb(pgid: int) -> float:
        total_kb = 0
        proc = Path("/proc")
        if not proc.is_dir():
            return 0.0
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="ascii", errors="replace")
                remainder = stat.rpartition(")")[2].split()
                if len(remainder) < 22 or int(remainder[2]) != pgid:
                    continue
                total_kb += int(remainder[21])
            except (OSError, ValueError, IndexError):
                continue
        return total_kb / 1024

    class _MemoryWatchdog(threading.Thread):
        def __init__(self, pgid: int, limit_mb: int):
            super().__init__(daemon=True)
            self._pgid = pgid
            self._limit_mb = limit_mb
            self._stop = threading.Event()
            self.exceeded = False

        def run(self) -> None:
            while not self._stop.wait(0.05):
                if _group_rss_mb(self._pgid) > self._limit_mb:
                    self.exceeded = True
                    try:
                        os.killpg(self._pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return

        def stop(self) -> None:
            self._stop.set()
            self.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _decode(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", errors="replace")


def _parse_report(stdout: str, nonce: str) -> dict | None:
    prefix = _RESULT_MARKER + nonce + " "
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                return None
    return None


def _interpreter() -> str:
    """Return an interpreter that does not need to spawn a child to start.

    On Windows a venv's ``python.exe`` is a launcher for the base interpreter;
    under a job object with ``ActiveProcessLimit=1`` that launcher cannot
    create its child. The harness only needs the standard library, so use the
    base interpreter directly.
    """
    if sys.platform == "win32" and sys.prefix != sys.base_prefix:
        candidate = Path(sys.base_prefix) / "python.exe"
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _kill(proc: subprocess.Popen, job) -> None:
    if sys.platform == "win32" and job:
        job.terminate()
    elif os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        proc.kill()


def run_python(
    source: str,
    tests_source: str,
    *,
    limits: SandboxLimits | None = None,
) -> SandboxResult:
    """Run ``source`` against ``tests_source`` in a fresh sandbox.

    ``tests_source`` must define ``test_*`` functions and may import
    ``solution`` (the candidate source under the fixed module name).
    """
    limits = limits or SandboxLimits()
    limits.validate()
    if not isinstance(source, str) or not isinstance(tests_source, str):
        raise TypeError("source and tests_source must be strings")
    if len(source.encode("utf-8")) > limits.max_source_bytes:
        raise ValueError("source exceeds max_source_bytes")
    if len(tests_source.encode("utf-8")) > limits.max_source_bytes:
        raise ValueError("tests_source exceeds max_source_bytes")

    with tempfile.TemporaryDirectory(prefix="gem-sandbox-") as root:
        root_path = Path(root)
        (root_path / "solution.py").write_text(source + "\n", encoding="utf-8")
        (root_path / "acceptance_tests.py").write_text(tests_source + "\n", encoding="utf-8")
        (root_path / "_harness.py").write_text(_HARNESS, encoding="utf-8")

        command = [
            _interpreter(),
            "-I",
            "-S",
            "-B",
            str(root_path / "_harness.py"),
            "1" if limits.allow_network else "0",
        ]
        kwargs: dict = {
            "cwd": root,
            "env": _build_env(root),
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        job = None
        watchdog = None
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["preexec_fn"] = _posix_preexec(limits)

        nonce = secrets.token_hex(16)
        started = time.monotonic()
        proc = subprocess.Popen(command, **kwargs)
        try:
            if sys.platform == "win32":
                job = _create_job(limits)
                _assign_job(job, proc)
            elif os.name == "posix":
                watchdog = _MemoryWatchdog(proc.pid, limits.memory_mb)
                watchdog.start()

            assert proc.stdin is not None
            proc.stdin.write(f"go {nonce}\n".encode("ascii"))
            proc.stdin.flush()

            timed_out = False
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=limits.wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(proc, job)
                stdout_bytes, stderr_bytes = proc.communicate()
        except BaseException:
            _kill(proc, job)
            raise
        finally:
            if watchdog is not None:
                watchdog.stop()
            if job:
                job.close()
            elif os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            if proc.poll() is None:
                proc.kill()
                proc.wait()

        duration = time.monotonic() - started
        stdout = _decode(stdout_bytes)
        stderr = _decode(stderr_bytes)
        report = _parse_report(stdout, nonce)
        memory_exceeded = bool(watchdog and watchdog.exceeded)
        tests: list[TestOutcome] = []

        if timed_out:
            status, error = "timeout", f"wall-clock limit of {limits.wall_seconds}s exceeded"
        elif memory_exceeded:
            status, error = "killed", f"memory limit of {limits.memory_mb} MiB exceeded"
        elif report is not None:
            tests = [
                TestOutcome(
                    name=str(item.get("name")),
                    passed=bool(item.get("passed")),
                    error=item.get("error"),
                )
                for item in report.get("tests", [])
            ]
            import_error = report.get("error")
            failed = [test for test in tests if not test.passed]
            if import_error:
                status, error = "error", str(import_error)
            elif not tests:
                status, error = "error", "no acceptance tests were found"
            elif failed:
                status, error = "failed", f"{len(failed)} of {len(tests)} tests failed"
            else:
                status, error = "passed", None
        else:
            status = "killed"
            error = (
                f"process exited with code {proc.returncode} without reporting a result"
                " (a missing or forged report is never accepted as success)"
            )
            if os.name == "posix" and proc.returncode is not None and proc.returncode < 0:
                error = f"process killed by signal {-proc.returncode} (likely a resource limit)"
            if stderr.strip():
                error = f"{error}: {stderr.strip().splitlines()[-1][:300]}"

        return SandboxResult(
            status=status,
            exit_code=proc.returncode,
            duration_seconds=duration,
            stdout=stdout,
            stderr=stderr,
            tests=tests if report is not None and not timed_out else [],
            error=error,
            isolation=_isolation_report(limits),
        )
