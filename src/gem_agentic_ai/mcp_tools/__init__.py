"""MCP tool server and the client bridge that exposes it to the agent.

The server is a real Model Context Protocol server over stdio; the bridge
discovers its tools with ``tools/list`` and invokes them with ``tools/call``.
See :mod:`gem_agentic_ai.mcp_tools.server` and
:mod:`gem_agentic_ai.mcp_tools.client`.
"""

from gem_agentic_ai.mcp_tools.client import (
    PROTOCOL_VERSION,
    MCPTool,
    MCPToolError,
    MCPToolset,
    ProtocolMismatchError,
    ServerConfig,
    args_model_for,
)
from gem_agentic_ai.mcp_tools.server import SERVER_NAME, SERVER_VERSION, build_server

__all__ = [
    "PROTOCOL_VERSION",
    "SERVER_NAME",
    "SERVER_VERSION",
    "MCPTool",
    "MCPToolError",
    "MCPToolset",
    "ProtocolMismatchError",
    "ServerConfig",
    "args_model_for",
    "build_server",
]
