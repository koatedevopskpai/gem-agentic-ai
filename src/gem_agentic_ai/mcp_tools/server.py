"""A small, real Model Context Protocol (MCP) tool server.

The server exposes two read-only tools over MCP's stdio transport:

* ``search_files`` - case-insensitive substring search over text files;
* ``read_file`` - read one text file.

Both tools are confined to an allow-listed root directory chosen when the
server starts. A path that escapes the root, a missing file, a binary file, a
file above the size limit and an empty search query are all refused as MCP
tool errors (``isError: true`` with a message the model can act on), not as
crashes. Unexpected exceptions are reported by the SDK as a generic tool
error, with the traceback kept on the server's stderr.

The SDK is pinned in ``requirements.txt``. The stdio handshake negotiates the
protocol revision recorded in :data:`gem_agentic_ai.mcp_tools.client.PROTOCOL_VERSION`.
Nothing here writes to stdout except the MCP framing produced by the SDK; a
stray ``print`` would corrupt the protocol stream, so diagnostics go to
stderr (the SDK logs there).

Run it directly with::

    python -m gem_agentic_ai.mcp_tools --root /path/to/allow-listed/dir

The transport is MCP over stdio: newline-delimited JSON-RPC 2.0 messages, as
specified for the stdio transport. It is not a plain HTTP endpoint.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

SERVER_NAME = "gem-agentic-ai-readonly"
SERVER_VERSION = "0.1.0"

#: Files above this size are skipped by search and refused by read.
MAX_FILE_BYTES = 1_000_000
#: Upper bound on how many files one search walks.
MAX_SCAN_FILES = 2_000
#: How much of a file search reads to decide whether it is binary.
BINARY_SNIFF_BYTES = 8_192
#: Longest line echoed in a search match, so one minified file cannot fill
#: the model's context by itself.
MAX_LINE_CHARS = 200

EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "htmlcov",
        "var",
    }
)


def _confined_path(root: Path, relative: str, *, what: str = "path") -> Path:
    """Resolve ``relative`` under ``root``, refusing anything that escapes.

    The check is on the resolved path, so a symlink cannot be used to step
    outside the allowed root either.
    """
    if not isinstance(relative, str) or not relative.strip():
        raise ToolError(f"{what} must be a non-empty relative path")
    if "\x00" in relative:
        raise ToolError(f"{what} contains a NUL byte")
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive:
        raise ToolError(f"{what} must be relative to the allowed root, not {relative!r}")
    if ".." in candidate.parts:
        raise ToolError(f"{what} may not contain '..': {relative!r}")
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ToolError(f"{what} resolves outside the allowed root: {relative!r}")
    return resolved


def _iter_files(root: Path, pattern: str) -> Iterator[Path]:
    """Yield the regular files under ``root`` matching a relative glob."""
    if not isinstance(pattern, str) or not pattern.strip():
        raise ToolError("glob_pattern must be a non-empty relative glob")
    candidate = Path(pattern)
    if candidate.is_absolute() or candidate.drive:
        raise ToolError("glob_pattern must be relative to the allowed root")
    if ".." in candidate.parts:
        raise ToolError("glob_pattern may not contain '..'")
    try:
        paths = sorted(root.glob(pattern))
    except (ValueError, NotImplementedError) as exc:
        raise ToolError(f"invalid glob_pattern {pattern!r}: {exc}") from exc

    for path in paths:
        try:
            if not path.is_file():
                continue
            resolved = path.resolve()
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            # A symlink that points outside the root, or a file that vanished
            # while the tree was being walked, is skipped rather than failing
            # the whole search.
            continue
        if EXCLUDED_DIRS.intersection(relative.parts[:-1]):
            continue
        yield path


def _read_text(path: Path) -> str | None:
    """Return the file's text, or ``None`` when it is binary or unreadable."""
    try:
        content = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in content[:BINARY_SNIFF_BYTES]:
        return None
    return content.decode("utf-8", errors="replace")


def build_server(root: Path, *, version: str = SERVER_VERSION) -> MCPServer:
    """Build the MCP server, confined to ``root``.

    ``root`` is resolved once, at construction; every tool call is checked
    against that resolved path.
    """
    bound_root = Path(root).expanduser().resolve()
    if not bound_root.is_dir():
        raise ValueError(f"root is not a directory: {bound_root}")

    server = MCPServer(
        SERVER_NAME,
        version=version,
        instructions=(
            "Read-only tools over one allow-listed directory. Paths are "
            "relative to that directory; absolute paths and '..' are refused."
        ),
    )

    @server.tool(
        title="Search files",
        description=(
            "Case-insensitive substring search across text files under the allowed "
            "root. Returns matching relative paths, line numbers and trimmed line "
            "text. Binary files, files above 1 MB and generated or dependency "
            "directories are skipped."
        ),
    )
    def search_files(
        query: str,
        glob_pattern: str = "**/*",
        max_results: Annotated[int, Field(ge=1, le=200)] = 50,
    ) -> dict[str, Any]:
        """Case-insensitive substring search across text files under the root.

        Returns matching files with line numbers. Binary files, files above
        1 MB and well-known generated or dependency directories are skipped.
        """
        needle = query.strip().lower()
        if not needle:
            raise ToolError("query must not be empty")
        if len(needle) > 200:
            raise ToolError("query must be at most 200 characters")

        matches: list[dict[str, Any]] = []
        files_scanned = 0
        files_skipped = 0
        truncated = False
        for path in _iter_files(bound_root, glob_pattern):
            if files_scanned >= MAX_SCAN_FILES:
                truncated = True
                break
            files_scanned += 1
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    files_skipped += 1
                    continue
            except OSError:
                files_skipped += 1
                continue
            text = _read_text(path)
            if text is None:
                files_skipped += 1
                continue
            relative = path.resolve().relative_to(bound_root).as_posix()
            for line_number, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    matches.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "text": line.strip()[:MAX_LINE_CHARS],
                        }
                    )
                    if len(matches) >= max_results:
                        truncated = True
                        break
            if truncated:
                break

        return {
            "matches": matches,
            "truncated": truncated,
            "files_scanned": files_scanned,
            "files_skipped": files_skipped,
        }

    @server.tool(
        title="Read file",
        description=(
            "Read one UTF-8 text file under the allowed root. The response says "
            "whether the text was truncated. Binary files, files above 1 MB and "
            "paths outside the root are refused."
        ),
    )
    def read_file(path: str, max_bytes: Annotated[int, Field(ge=1, le=100_000)] = 20_000) -> dict:
        """Read one UTF-8 text file under the root.

        ``truncated`` is true when the file is longer than ``max_bytes``.
        Binary files, files above 1 MB and paths outside the root are refused.
        """
        resolved = _confined_path(bound_root, path)
        if not resolved.exists():
            raise ToolError(f"file not found: {path}")
        if not resolved.is_file():
            raise ToolError(f"not a regular file: {path}")
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise ToolError(f"cannot stat {path}: {exc}") from exc
        if size > MAX_FILE_BYTES:
            raise ToolError(f"file is {size} bytes; the limit is {MAX_FILE_BYTES} bytes")
        try:
            with resolved.open("rb") as handle:
                content = handle.read(max_bytes + 1)
        except OSError as exc:
            raise ToolError(f"cannot read {path}: {exc}") from exc
        if b"\x00" in content[:BINARY_SNIFF_BYTES]:
            raise ToolError(f"file appears to be binary: {path}")
        truncated = len(content) > max_bytes
        return {
            "path": resolved.relative_to(bound_root).as_posix(),
            "text": content[:max_bytes].decode("utf-8", errors="replace"),
            "truncated": truncated,
            "size_bytes": size,
        }

    return server


def main(argv: list[str] | None = None) -> int:
    """Run the server over stdio. Diagnostics go to stderr, never stdout."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m gem_agentic_ai.mcp_tools")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Directory the tools may read (default: current directory)",
    )
    arguments = parser.parse_args(argv)
    server = build_server(arguments.root)
    server.run("stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
