"""Wire-level tests for the MCP stdio server.

The wire tests do not use the MCP SDK client. They spawn
``python -m gem_agentic_ai.mcp_tools`` exactly as a third-party MCP client
would and write newline-delimited JSON-RPC 2.0 to its stdin, so the framing,
the handshake and the result envelopes are tested rather than assumed.

The last group drives the same server through the SDK's in-process transport.
Those tests are about the tool's own boundaries (absolute paths, escaped
globs, excluded directories, result limits), not about framing.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from gem_agentic_ai.mcp_tools.server import build_server

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "2025-11-25"


class WireServer:
    """A running MCP server subprocess, driven one JSON-RPC message at a time."""

    def __init__(self, root: Path) -> None:
        env = dict(os.environ)
        source = str(REPO_ROOT / "src")
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = source + (os.pathsep + existing if existing else "")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "gem_agentic_ai.mcp_tools", "--root", str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(REPO_ROOT),
            env=env,
        )
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: list[str] = []
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._next_id = 0

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._stdout.put(line)
        self._stdout.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line)

    def send(self, message: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def read(self, timeout: float = 15.0) -> dict:
        try:
            line = self._stdout.get(timeout=timeout)
        except queue.Empty:
            pytest.fail(
                "timed out waiting for an MCP response; server stderr:\n"
                + "".join(self._stderr[-20:])
            )
        if line is None:
            pytest.fail(
                "the MCP server closed stdout; server stderr:\n" + "".join(self._stderr[-20:])
            )
        return json.loads(line)

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        response = self.read()
        assert response.get("id") == self._next_id, response
        return response

    def notify(self, method: str, params: dict | None = None) -> None:
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def handshake(self) -> dict:
        response = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "wire-test", "version": "0.0.0"},
            },
        )
        self.notify("notifications/initialized")
        return response

    def close(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a hung server
            self.proc.kill()
            self.proc.wait(timeout=5)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "notes").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "notes" / "alpha.md").write_text(
        "# Alpha\n\nThe checkpoint keeps conversation state.\n", encoding="utf-8"
    )
    (root / "notes" / "beta.txt").write_text(
        "first line\nsecond checkpoint here\n", encoding="utf-8"
    )
    (root / "src" / "app.py").write_text("def main():\n    return 'checkpoint'\n", encoding="utf-8")
    (root / "notes" / "blob.bin").write_bytes(b"\x00\x01binary checkpoint\x00")
    (tmp_path / "outside.txt").write_text("checkpoint outside the root\n", encoding="utf-8")
    return root


@pytest.fixture()
def wire(workspace: Path):
    server = WireServer(workspace)
    try:
        yield server
    finally:
        server.close()


def text_of(result: dict) -> str:
    content = result.get("content") or []
    return "\n".join(item.get("text", "") for item in content if item.get("type") == "text")


def test_handshake_negotiates_the_pinned_revision_and_advertises_tools(wire: WireServer) -> None:
    response = wire.handshake()

    result = response["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "gem-agentic-ai-readonly"
    assert "tools" in result["capabilities"]


def test_tools_list_describes_both_tools_with_json_schemas(wire: WireServer) -> None:
    wire.handshake()

    result = wire.request("tools/list")["result"]

    tools = {tool["name"]: tool for tool in result["tools"]}
    assert set(tools) == {"search_files", "read_file"}
    search = tools["search_files"]["inputSchema"]
    assert search["type"] == "object"
    assert search["required"] == ["query"]
    assert search["properties"]["max_results"]["maximum"] == 200
    assert tools["read_file"]["inputSchema"]["required"] == ["path"]


def test_unused_capabilities_are_advertised_but_empty(wire: WireServer) -> None:
    """The SDK advertises its default capabilities; nothing backs prompts/resources."""

    wire.handshake()

    prompts = wire.request("prompts/list")["result"]
    resources = wire.request("resources/list")["result"]

    assert prompts["prompts"] == []
    assert resources["resources"] == []


def test_tools_call_returns_matches_over_the_protocol(wire: WireServer) -> None:
    wire.handshake()

    response = wire.request(
        "tools/call",
        {"name": "search_files", "arguments": {"query": "checkpoint"}},
    )

    result = response["result"]
    assert result.get("isError") is not True
    payload = json.loads(text_of(result))
    paths = {match["path"] for match in payload["matches"]}
    assert {"notes/alpha.md", "notes/beta.txt", "src/app.py"}.issubset(paths)
    # The binary file is skipped rather than decoded.
    assert "notes/blob.bin" not in paths


def test_unknown_tool_is_a_tool_error(wire: WireServer) -> None:
    wire.handshake()

    response = wire.request("tools/call", {"name": "does_not_exist", "arguments": {}})

    result = response["result"]
    assert result["isError"] is True
    assert "Unknown tool" in text_of(result)


def test_invalid_arguments_are_a_tool_error(wire: WireServer) -> None:
    wire.handshake()

    response = wire.request(
        "tools/call",
        {"name": "search_files", "arguments": {"query": "anything", "max_results": 10_000}},
    )

    result = response["result"]
    assert result["isError"] is True
    assert "max_results" in text_of(result) or "less than or equal" in text_of(result)


def test_a_path_escape_is_refused_over_the_protocol(wire: WireServer) -> None:
    wire.handshake()

    response = wire.request(
        "tools/call",
        {"name": "read_file", "arguments": {"path": "../outside.txt"}},
    )

    result = response["result"]
    assert result["isError"] is True
    assert "may not contain '..'" in text_of(result)


def test_unknown_method_is_a_json_rpc_error(wire: WireServer) -> None:
    wire.handshake()

    response = wire.request("bogus/method", {})

    assert "result" not in response
    assert response["error"]["code"] == -32601


def inproc_call(root: Path, name: str, arguments: dict) -> Any:
    """Call the server through the SDK's in-process transport."""

    async def scenario() -> Any:
        async with Client(build_server(root)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(scenario())


def test_absolute_paths_are_refused(workspace: Path) -> None:
    result = inproc_call(workspace, "read_file", {"path": str(workspace / "notes" / "alpha.md")})

    assert result.is_error is True
    assert "must be relative" in result.content[0].text


def test_an_escaped_glob_is_refused(workspace: Path) -> None:
    result = inproc_call(workspace, "search_files", {"query": "checkpoint", "glob_pattern": "../*"})

    assert result.is_error is True
    assert "may not contain '..'" in result.content[0].text


def test_generated_schema_constraints_reject_bad_arguments(workspace: Path) -> None:
    result = inproc_call(workspace, "search_files", {"query": "checkpoint", "max_results": 0})

    assert result.is_error is True
    assert "max_results" in result.content[0].text


def test_excluded_directories_are_not_searched(workspace: Path) -> None:
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("checkpoint in git metadata\n", encoding="utf-8")

    result = inproc_call(workspace, "search_files", {"query": "checkpoint"})

    payload = json.loads(result.content[0].text)
    assert all(not match["path"].startswith(".git/") for match in payload["matches"])


def test_search_honours_max_results_and_reports_truncation(workspace: Path) -> None:
    result = inproc_call(
        workspace, "search_files", {"query": "checkpoint", "max_results": 1}
    )

    payload = json.loads(result.content[0].text)
    assert len(payload["matches"]) == 1
    assert payload["truncated"] is True


def test_build_server_rejects_a_root_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        build_server(tmp_path / "missing")


def test_the_server_keeps_serving_after_a_tool_error(wire: WireServer) -> None:
    wire.handshake()
    first = wire.request("tools/call", {"name": "does_not_exist", "arguments": {}})
    assert first["result"]["isError"] is True

    second = wire.request(
        "tools/call",
        {"name": "read_file", "arguments": {"path": "notes/alpha.md"}},
    )

    assert second["result"].get("isError") is not True
    assert "checkpoint keeps conversation state" in text_of(second["result"])
