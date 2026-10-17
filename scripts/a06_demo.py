"""End-to-end demo of the MCP tool server, the bridge and the agent.

The demo:
  1. creates a small workspace and an outside file the tools must not reach;
  2. launches the real MCP server as a subprocess and performs the protocol
     handshake, printing the negotiated revision and the discovered tools;
  3. calls both tools directly over the shared session;
  4. exercises unknown tools, a path escape and an empty query, and checks
     that each is an error rather than a silent success;
  5. runs the LangGraph agent with a scripted model that asks for a search,
     proving the agent discovers and invokes the tool through the protocol;
  6. runs the agent again with a call that the server refuses, proving the
     error reaches the model as an error ToolMessage instead of ending the
     run.

No network or model provider is used. Exit status is 0 only if every check
passes. Run from the repository root:

    python scripts/a06_demo.py --out artifacts/a06_demo.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gem_agentic_ai.agent.graph import build_agent  # noqa: E402
from gem_agentic_ai.mcp_tools import (  # noqa: E402
    MCPToolError,
    MCPToolset,
    ServerConfig,
)


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """Replay fixed AIMessages; record every context the agent supplied."""

    responses: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._cursor >= len(self.responses):
            raise AssertionError("the scripted model ran out of responses")
        self.seen.append(list(messages))
        response = self.responses[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self


def build_workspace(base: Path) -> tuple[Path, Path]:
    root = base / "workspace"
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
    outside = base / "outside.txt"
    outside.write_text("checkpoint outside the root\n", encoding="utf-8")
    return root, outside


async def run_checks(root: Path, outside: Path) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"checks": checks}
    config = ServerConfig(root=root, env={"PYTHONPATH": str(SRC)})

    async with MCPToolset(config) as toolset:
        report["protocol"] = toolset.protocol_version
        report["server"] = {
            "name": toolset.server_info.name,
            "version": toolset.server_info.version,
        }
        report["tools"] = {
            tool.name: {
                "required": tool.args_schema.model_json_schema().get("required", []),
                "description": tool.description,
            }
            for tool in toolset.tools
        }
        print(f"protocol: {toolset.protocol_version} (pinned)")
        print(f"server: {toolset.server_info.name} {toolset.server_info.version}")
        print(f"tools: {', '.join(tool.name for tool in toolset.tools)}")

        started = time.perf_counter()
        search = json.loads(await toolset.call_tool("search_files", {"query": "checkpoint"}))
        search_ms = (time.perf_counter() - started) * 1000
        paths = {match["path"] for match in search["matches"]}
        checks["search"] = {
            "matched": sorted(paths),
            "binary_skipped": "notes/blob.bin" not in paths,
            "files_scanned": search["files_scanned"],
            "elapsed_ms": round(search_ms, 2),
            "passed": len(search["matches"]) >= 3 and "notes/blob.bin" not in paths,
        }

        read = json.loads(await toolset.call_tool("read_file", {"path": "notes/alpha.md"}))
        checks["read"] = {
            "path": read["path"],
            "has_text": "conversation state" in read["text"],
            "passed": read["path"] == "notes/alpha.md" and "conversation state" in read["text"],
        }

        try:
            await toolset.call_tool("does_not_exist", {})
        except MCPToolError as exc:
            checks["unknown_tool"] = {"error": str(exc), "passed": "Unknown tool" in str(exc)}
        else:
            checks["unknown_tool"] = {"error": None, "passed": False}

        try:
            await toolset.call_tool("read_file", {"path": "../outside.txt"})
        except MCPToolError as exc:
            checks["path_escape"] = {"error": str(exc), "passed": "may not contain" in str(exc)}
        else:
            checks["path_escape"] = {"error": None, "passed": False}

        try:
            await toolset.call_tool("search_files", {"query": "  "})
        except MCPToolError as exc:
            checks["empty_query"] = {
                "error": str(exc),
                "passed": "query must not be empty" in str(exc),
            }
        else:
            checks["empty_query"] = {"error": None, "passed": False}

        # Agent: discovery -> tools/list -> tools/call -> observation -> answer.
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
        started = time.perf_counter()
        result = agent.invoke(
            {"messages": [("user", "Where is the checkpoint discussed?")]},
            {"configurable": {"thread_id": "a06-demo"}},
        )
        agent_ms = (time.perf_counter() - started) * 1000
        tool_messages = [
            message for message in result["messages"] if isinstance(message, ToolMessage)
        ]
        model_saw_matches = bool(
            tool_messages and "notes/alpha.md" in model.seen[1][-1].content
        )
        checks["agent"] = {
            "tool": tool_messages[0].name if tool_messages else None,
            "tool_status": tool_messages[0].status if tool_messages else None,
            "tool_result_has_match": bool(
                tool_messages and "notes/alpha.md" in tool_messages[0].content
            ),
            "model_saw_tool_result": model_saw_matches,
            "final_answer": result["messages"][-1].content,
            "elapsed_ms": round(agent_ms, 2),
            "budget_records": [
                {"tool": record.tool, "status": str(record.status)}
                for record in agent.budget.records
            ],
            "passed": bool(
                tool_messages
                and tool_messages[0].name == "search_files"
                and tool_messages[0].status == "success"
                and model_saw_matches
                and any(record.status == "ok" for record in agent.budget.records)
            ),
        }

        # Agent: a tool error becomes an error ToolMessage, not a crash.
        error_model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[tool_call("search_files", {"query": " "}, call_id="call_err")],
                ),
                AIMessage(content="The search was refused, so I have no matches."),
            ]
        )
        error_agent = build_agent(error_model, list(toolset.tools))
        error_result = error_agent.invoke(
            {"messages": [("user", "Search for nothing")]},
            {"configurable": {"thread_id": "a06-demo-error"}},
        )
        error_tool_messages = [
            message for message in error_result["messages"] if isinstance(message, ToolMessage)
        ]
        checks["agent_tool_error"] = {
            "tool_status": error_tool_messages[0].status if error_tool_messages else None,
            "model_saw_error": bool(
                error_tool_messages and "query must not be empty" in error_model.seen[1][-1].content
            ),
            "final_answer": error_result["messages"][-1].content,
            "passed": bool(
                error_tool_messages
                and error_tool_messages[0].status == "error"
                and "query must not be empty" in error_tool_messages[0].content
                and "query must not be empty" in error_model.seen[1][-1].content
            ),
        }

    # The outside file exists; nothing above reached it.
    report["outside_file_untouched"] = outside.exists()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP tool server demo")
    parser.add_argument("--out", type=Path, default=Path("artifacts/a06_demo.json"))
    arguments = parser.parse_args(argv)

    os.environ.setdefault("PYTHONPATH", str(SRC))
    with tempfile.TemporaryDirectory(prefix="a06-demo-") as temporary:
        root, outside = build_workspace(Path(temporary))
        report = asyncio.run(run_checks(root, outside))

    for name, check in report["checks"].items():
        detail = {key: value for key, value in check.items() if key != "passed"}
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{name}: {status} ({json.dumps(detail, default=str)})")
    overall = all(check["passed"] for check in report["checks"].values()) and report[
        "outside_file_untouched"
    ]
    report["overall"] = "PASS" if overall else "FAIL"
    print(f"overall: {report['overall']}")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
