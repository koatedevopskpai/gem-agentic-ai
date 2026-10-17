"""MCP client bridge: discover tools over ``tools/list`` and expose them to
the agent as LangChain tools that call ``tools/call``.

The bridge is deliberately thin, but it is not a mock of MCP: it launches the
real server as a subprocess and speaks the protocol's newline-delimited
JSON-RPC over stdin/stdout. Two surfaces are offered:

* :class:`MCPToolset` - an async context manager holding one live session.
  Use it to list tools, inspect schemas, or call a tool directly.
* :class:`MCPTool` - a LangChain ``BaseTool`` built from one MCP tool's
  ``inputSchema``. The agent's ``ToolGuard`` requires a Pydantic argument
  model, so the JSON Schema is converted to one. An unsupported schema is
  refused at discovery time rather than exposed without validation.

Errors are not swallowed. A ``tools/call`` result with ``isError: true`` is
raised as :class:`MCPToolError` with the server's message. The guard turns
that into a ``ToolMessage`` whose status is ``error``, which is what the
model sees; the run does not end.

The protocol revision is pinned. If the server negotiates a different
revision, the bridge refuses to continue rather than guess at compatibility.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, Tool
from pydantic import ConfigDict, Field, create_model

#: The MCP revision this bridge pins. ``mcp==2.3.0`` (see requirements.txt)
#: still supports the initialize handshake, and with ``mode="legacy"``
#: negotiates this revision.
PROTOCOL_VERSION = "2025-11-25"

_JSON_SCALARS: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"$ref", "anyOf", "oneOf", "allOf", "not", "patternProperties"}
)


class MCPToolError(RuntimeError):
    """A tool call failed. The message is the server's, not a substitute."""


class ProtocolMismatchError(MCPToolError):
    """The server negotiated a revision this bridge does not speak."""


@dataclass(frozen=True)
class ServerConfig:
    """How to launch the MCP server and which root it may read."""

    root: Path
    command: str = sys.executable
    args: tuple[str, ...] = ("-m", "gem_agentic_ai.mcp_tools")
    env: Mapping[str, str] | None = None
    protocol_version: str = PROTOCOL_VERSION

    def parameters(self) -> StdioServerParameters:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise MCPToolError(f"MCP server root is not a directory: {root}")
        return StdioServerParameters(
            command=self.command,
            args=[*self.args, "--root", str(root)],
            env=dict(self.env) if self.env is not None else None,
        )


def _check_protocol(actual: str | None, expected: str) -> None:
    if actual != expected:
        raise ProtocolMismatchError(
            f"MCP server negotiated protocol {actual!r}; this bridge pins {expected!r}"
        )


def _result_text(result: CallToolResult) -> str:
    """The text the model should see for a tool result."""
    chunks = [item.text for item in result.content if getattr(item, "type", None) == "text"]
    if chunks:
        return "\n".join(chunks)
    if result.structured_content is not None:
        return json.dumps(result.structured_content, indent=2, sort_keys=True)
    return ""


async def _invoke(client: Client, name: str, arguments: Mapping[str, Any]) -> str:
    result = await client.call_tool(name, dict(arguments))
    text = _result_text(result)
    if result.is_error:
        detail = text or "the server returned an error with no message"
        raise MCPToolError(f"MCP tool '{name}' failed: {detail}")
    return text


def _scalar_type(schema: Mapping[str, Any], *, field_name: str, tool_name: str) -> Any:
    schema_type = schema.get("type")
    if schema_type not in _JSON_SCALARS:
        raise MCPToolError(
            f"tool '{tool_name}': field '{field_name}' has unsupported JSON Schema type "
            f"{schema_type!r}; refusing to expose it without validation"
        )
    return _JSON_SCALARS[schema_type]


def _field_from_schema(
    schema: Mapping[str, Any], *, field_name: str, tool_name: str, required: bool
) -> tuple[Any, Any]:
    unsupported = _UNSUPPORTED_SCHEMA_KEYS.intersection(schema)
    if unsupported:
        raise MCPToolError(
            f"tool '{tool_name}': field '{field_name}' uses unsupported JSON Schema "
            f"keyword(s) {sorted(unsupported)}; refusing to expose it without validation"
        )

    constraints: dict[str, Any] = {}
    if "enum" in schema:
        values = schema["enum"]
        if not values or not all(isinstance(value, (str, int, bool)) for value in values):
            raise MCPToolError(
                f"tool '{tool_name}': field '{field_name}' has an enum this bridge "
                "cannot represent"
            )
        value_type: Any = Literal[tuple(values)]  # type: ignore[valid-type]
    else:
        schema_type = schema.get("type")
        if schema_type == "array":
            item_type = _scalar_type(
                schema.get("items", {}), field_name=f"{field_name}[]", tool_name=tool_name
            )
            value_type = list[item_type]  # type: ignore[valid-type]
        elif schema_type == "object":
            value_type = dict[str, Any]
        else:
            value_type = _scalar_type(schema, field_name=field_name, tool_name=tool_name)

    # Constraint keywords are carried across only where JSON Schema and
    # Pydantic agree on their meaning.
    for source, target in (
        ("minimum", "ge"),
        ("maximum", "le"),
        ("minLength", "min_length"),
        ("maxLength", "max_length"),
    ):
        if source in schema:
            constraints[target] = schema[source]

    if required:
        return value_type, Field(..., **constraints)
    if "default" in schema:
        return value_type, Field(default=schema["default"], **constraints)
    return value_type | None, Field(default=None, **constraints)


def args_model_for(tool: Tool) -> type:
    """Build a Pydantic argument model from an MCP tool's ``inputSchema``.

    The guard refuses a tool without a Pydantic ``args_schema``, so this
    conversion is what makes an MCP tool callable by the agent at all.
    Anything this bridge cannot validate is refused here, at discovery.
    """
    schema = tool.input_schema or {}
    if schema.get("type") != "object":
        raise MCPToolError(f"tool '{tool.name}': inputSchema is not an object schema")

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, tuple[Any, Any]] = {}
    for name, property_schema in properties.items():
        if not isinstance(property_schema, Mapping):
            raise MCPToolError(f"tool '{tool.name}': property '{name}' is not an object schema")
        fields[name] = _field_from_schema(
            property_schema,
            field_name=name,
            tool_name=tool.name,
            required=name in required,
        )

    model_name = "".join(character if character.isalnum() else "_" for character in tool.name)
    if not model_name or model_name[0].isdigit():
        model_name = f"Tool_{model_name}"
    return create_model(f"{model_name}Arguments", **fields)  # type: ignore[call-overload]


class MCPTool(BaseTool):
    """A LangChain tool whose implementation is an MCP ``tools/call``.

    When ``session`` is set (by :class:`MCPToolset`) the call uses that live
    session. Otherwise ``_run`` opens a short-lived stdio session for the
    call. The second path is the one the agent uses: ``ToolGuard`` executes
    every tool call on a worker thread with a timeout, so the tool cannot
    borrow a session owned by another thread's event loop. One session per
    call keeps that boundary honest at the cost of a process spawn.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    server: ServerConfig
    session: Any = None

    def _payload(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        # Optional fields the guard materialised as None are omitted so the
        # server applies its own documented defaults.
        return {key: value for key, value in kwargs.items() if value is not None}

    def _run(self, **kwargs: Any) -> str:
        return asyncio.run(self._call_once(self._payload(kwargs)))

    async def _arun(self, **kwargs: Any) -> str:
        if self.session is None:
            return await self._call_once(self._payload(kwargs))
        return await _invoke(self.session, self.name, self._payload(kwargs))

    async def _call_once(self, arguments: Mapping[str, Any]) -> str:
        failure: MCPToolError | None = None
        async with Client(self.server.parameters(), mode="legacy") as client:
            try:
                _check_protocol(client.protocol_version, self.server.protocol_version)
                return await _invoke(client, self.name, arguments)
            except MCPToolError as exc:
                # Raised after the session has closed. Letting it travel
                # through the client's AnyIO task group would wrap it in an
                # ExceptionGroup and bury the server's message.
                failure = exc
        assert failure is not None
        raise failure


@dataclass
class MCPToolset:
    """One live MCP session and the LangChain tools discovered from it.

    Use as an async context manager::

        async with MCPToolset(ServerConfig(root)) as toolset:
            names = [tool.name for tool in toolset.tools]
            answer = await toolset.call_tool("search_files", {"query": "needle"})
            agent = build_agent(llm, toolset.tools)
    """

    config: ServerConfig
    protocol_version: str | None = None
    server_info: Any = None
    tools: list[MCPTool] = field(default_factory=list)
    _stack: AsyncExitStack | None = field(default=None, repr=False)
    _client: Client | None = field(default=None, repr=False)

    async def __aenter__(self) -> MCPToolset:
        stack = AsyncExitStack()
        try:
            parameters = self.config.parameters()
            client = await stack.enter_async_context(Client(parameters, mode="legacy"))
            _check_protocol(client.protocol_version, self.config.protocol_version)
            listed = await client.list_tools()
        except BaseException:
            await stack.aclose()
            raise

        self._stack = stack
        self._client = client
        self.protocol_version = client.protocol_version
        self.server_info = client.server_info
        self.tools = [
            MCPTool(
                name=tool.name,
                description=tool.description or tool.name,
                args_schema=args_model_for(tool),
                server=self.config,
                session=client,
            )
            for tool in listed.tools
        ]
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        stack, self._stack = self._stack, None
        self._client = None
        if stack is not None:
            await stack.aclose()

    @property
    def client(self) -> Client:
        if self._client is None:
            raise MCPToolError("MCPToolset is not connected; use it as an async context manager")
        return self._client

    def tool(self, name: str) -> MCPTool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(f"no MCP tool named {name!r}; discovered: {[t.name for t in self.tools]}")

    async def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> str:
        return await _invoke(self.client, name, arguments or {})
