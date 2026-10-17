"""Tests for the MCP client bridge and its use by the agent.

Every test launches the real server as a subprocess and talks MCP to it. The
network is not used, so the tests run offline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from mcp.types import Tool

from gem_agentic_ai.agent.graph import build_agent
from gem_agentic_ai.agent.simple import run_simple_agent
from gem_agentic_ai.bounds import CallStatus
from gem_agentic_ai.mcp_tools import (
    PROTOCOL_VERSION,
    MCPToolError,
    MCPToolset,
    ProtocolMismatchError,
    ServerConfig,
    args_model_for,
)
from tests.fakes import ScriptedChatModel, tool_call

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def config_for(root: Path, **overrides: object) -> ServerConfig:
    values: dict[str, object] = {
        "root": root,
        "env": {"PYTHONPATH": str(REPO_ROOT / "src")},
    }
    values.update(overrides)
    return ServerConfig(**values)


def test_discovery_reports_the_pinned_protocol_and_both_tools(workspace: Path) -> None:
    async def scenario() -> tuple[str | None, str, dict]:
        async with MCPToolset(config_for(workspace)) as toolset:
            return (
                toolset.protocol_version,
                toolset.server_info.name,
                {tool.name: tool.args_schema.model_json_schema() for tool in toolset.tools},
            )

    protocol, server_name, schemas = asyncio.run(scenario())

    assert protocol == PROTOCOL_VERSION
    assert server_name == "gem-agentic-ai-readonly"
    assert set(schemas) == {"search_files", "read_file"}
    assert schemas["search_files"]["properties"]["max_results"]["maximum"] == 200
    assert schemas["read_file"]["required"] == ["path"]


def test_a_mismatched_protocol_is_refused_and_the_server_is_stopped(workspace: Path) -> None:
    async def scenario() -> None:
        async with MCPToolset(config_for(workspace, protocol_version="2025-06-18")):
            pytest.fail("the bridge accepted a protocol revision it does not pin")

    with pytest.raises(ProtocolMismatchError, match="2025-06-18"):
        asyncio.run(scenario())


def test_search_returns_matches_over_the_shared_session(workspace: Path) -> None:
    async def scenario() -> dict:
        async with MCPToolset(config_for(workspace)) as toolset:
            text = await toolset.call_tool("search_files", {"query": "checkpoint"})
            return json.loads(text)

    payload = asyncio.run(scenario())

    paths = {match["path"] for match in payload["matches"]}
    assert {"notes/alpha.md", "notes/beta.txt", "src/app.py"}.issubset(paths)
    assert "notes/blob.bin" not in paths
    assert payload["truncated"] is False


def test_a_tool_can_be_awaited_on_the_shared_session(workspace: Path) -> None:
    async def scenario() -> str:
        async with MCPToolset(config_for(workspace)) as toolset:
            return await toolset.tool("read_file").ainvoke({"path": "notes/alpha.md"})

    text = asyncio.run(scenario())

    assert "checkpoint keeps conversation state" in text


def test_read_file_returns_text(workspace: Path) -> None:
    async def scenario() -> dict:
        async with MCPToolset(config_for(workspace)) as toolset:
            return json.loads(await toolset.call_tool("read_file", {"path": "notes/alpha.md"}))

    payload = asyncio.run(scenario())

    assert payload["path"] == "notes/alpha.md"
    assert "checkpoint keeps conversation state" in payload["text"]
    assert payload["truncated"] is False


def test_an_unknown_tool_raises_with_the_servers_message(workspace: Path) -> None:
    async def scenario() -> None:
        async with MCPToolset(config_for(workspace)) as toolset:
            await toolset.call_tool("does_not_exist", {})

    with pytest.raises(MCPToolError, match="Unknown tool: does_not_exist"):
        asyncio.run(scenario())


def test_a_tool_error_is_raised_not_swallowed(workspace: Path) -> None:
    async def scenario() -> None:
        async with MCPToolset(config_for(workspace)) as toolset:
            await toolset.call_tool("search_files", {"query": "   "})

    with pytest.raises(MCPToolError, match="query must not be empty"):
        asyncio.run(scenario())


def test_a_path_escape_is_refused(workspace: Path) -> None:
    async def scenario() -> None:
        async with MCPToolset(config_for(workspace)) as toolset:
            await toolset.call_tool("read_file", {"path": "../outside.txt"})

    with pytest.raises(MCPToolError, match="may not contain"):
        asyncio.run(scenario())


def test_a_binary_file_is_refused_by_read_and_skipped_by_search(workspace: Path) -> None:
    async def scenario() -> tuple[dict, str]:
        async with MCPToolset(config_for(workspace)) as toolset:
            matches = json.loads(await toolset.call_tool("search_files", {"query": "checkpoint"}))
            try:
                await toolset.call_tool("read_file", {"path": "notes/blob.bin"})
            except MCPToolError as exc:
                return matches, str(exc)
            raise AssertionError("a binary file was read")

    matches, error = asyncio.run(scenario())

    assert all(match["path"] != "notes/blob.bin" for match in matches["matches"])
    assert "binary" in error


def test_a_file_above_the_size_limit_is_refused(workspace: Path) -> None:
    (workspace / "notes" / "large.txt").write_bytes(b"x" * 1_000_001)

    async def scenario() -> None:
        async with MCPToolset(config_for(workspace)) as toolset:
            await toolset.call_tool("read_file", {"path": "notes/large.txt"})

    with pytest.raises(MCPToolError, match="the limit is"):
        asyncio.run(scenario())


def test_the_tool_runs_from_the_guards_worker_thread(workspace: Path) -> None:
    """``_run`` opens its own session, because the guard calls it in a thread."""

    async def build() -> object:
        async with MCPToolset(config_for(workspace)) as toolset:
            return toolset.tool("search_files")

    tool = asyncio.run(build())

    text = tool.invoke({"query": "conversation state"})

    assert "notes/alpha.md" in text


def test_enums_defaults_and_constraints_are_converted_to_a_model() -> None:
    tool = Tool.model_validate(
        {
            "name": "pick",
            "description": "uses an enum and a bound",
            "inputSchema": {
                "type": "object",
                "required": ["mode"],
                "properties": {
                    "mode": {"type": "string", "enum": ["text", "path"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
                },
            },
        }
    )

    model = args_model_for(tool)

    assert model(mode="text").limit == 3
    with pytest.raises(ValueError):
        model(mode="other")
    with pytest.raises(ValueError):
        model(mode="text", limit=6)


def test_an_unsupported_json_schema_is_refused_at_discovery() -> None:
    tool = Tool.model_validate(
        {
            "name": "refs",
            "description": "uses an unsupported construct",
            "inputSchema": {
                "type": "object",
                "properties": {"thing": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            },
        }
    )

    with pytest.raises(MCPToolError, match="unsupported JSON Schema"):
        args_model_for(tool)


def test_the_agent_invokes_the_tool_over_the_protocol(workspace: Path) -> None:
    async def scenario() -> tuple[object, ScriptedChatModel]:
        async with MCPToolset(config_for(workspace)) as toolset:
            model = ScriptedChatModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[tool_call("search_files", {"query": "checkpoint"})],
                    ),
                    AIMessage(content="The checkpoint note is in notes/alpha.md."),
                ]
            )
            agent = build_agent(model, list(toolset.tools))
            result = agent.invoke(
                {"messages": [("user", "Where is the checkpoint discussed?")]},
                {"configurable": {"thread_id": "mcp-agent"}},
            )
            return result, model

    result, model = asyncio.run(scenario())

    assert result["messages"][-1].content == "The checkpoint note is in notes/alpha.md."
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "search_files"
    assert tool_messages[0].status == "success"
    assert "notes/alpha.md" in tool_messages[0].content
    # The model's second turn saw the MCP result, not an empty placeholder.
    assert any(
        isinstance(message, ToolMessage) and "notes/alpha.md" in message.content
        for message in model.seen[1]
    )


def test_a_tool_error_reaches_the_model_as_an_error_message(workspace: Path) -> None:
    async def scenario() -> tuple[object, ScriptedChatModel]:
        async with MCPToolset(config_for(workspace)) as toolset:
            model = ScriptedChatModel(
                responses=[
                    AIMessage(content="", tool_calls=[tool_call("search_files", {"query": " "})]),
                    AIMessage(content="The search was refused."),
                ]
            )
            agent = build_agent(model, list(toolset.tools))
            result = agent.invoke(
                {"messages": [("user", "Search for nothing")]},
                {"configurable": {"thread_id": "mcp-agent-error"}},
            )
            return result, model

    result, model = asyncio.run(scenario())

    assert result["messages"][-1].content == "The search was refused."
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert tool_messages[0].status == "error"
    assert "query must not be empty" in tool_messages[0].content
    assert "query must not be empty" in model.seen[1][-1].content


def test_an_unknown_tool_is_refused_by_an_agent_registry(workspace: Path) -> None:
    async def scenario() -> tuple[object, ScriptedChatModel]:
        async with MCPToolset(config_for(workspace)) as toolset:
            model = ScriptedChatModel(
                responses=[
                    AIMessage(content="", tool_calls=[tool_call("read_file", {"path": "x"})]),
                    AIMessage(content="That tool is not available."),
                ]
            )
            result = run_simple_agent(
                model,
                "Read the file",
                {"search_files": toolset.tool("search_files")},
            )
            return result, model

    result, model = asyncio.run(scenario())

    assert result.answer == "That tool is not available."
    assert result.budget.records[0].status == CallStatus.REFUSED
    assert "unknown tool 'read_file'" in model.seen[1][-1].content
