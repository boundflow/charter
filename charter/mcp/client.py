"""Connecting to MCP servers and turning their tools into BoundFlow tools.

One MCP server is one process or endpoint exposing many tools. Charter connects,
calls `tools/list`, and keeps only what the agent config declares — the config is a
filter, not a description. Tools the server exposes but the config omits are never
shown to the model, so a server shipping a new tool in an upgrade cannot silently
widen an agent's authority.

The load-bearing detail is `_result_text`: MCP reports most tool errors as a
*successful* JSON-RPC response carrying `isError: true`. Returning that as a value
would make BoundFlow count a failure as a success — silently disabling
`on_failure`, `tool_failures` rules, and every failure metric at once. So it raises.
"""

from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from boundflow import Tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config.agent import AgentConfig, McpServer, split_qualified

log = logging.getLogger(__name__)


class McpError(Exception):
    """A tool call failed. Raised so BoundFlow counts it as a tool failure."""


class QuarantineError(Exception):
    """This agent can't run: a declared tool isn't there, or a server won't connect.

    Not fatal to the worker — one agent's broken server must not take down the
    others a worker serves. The agent is marked unhealthy and its tasks fail fast
    with this reason, which increments num_failures and lets the lifecycle rules
    pause it automatically.
    """


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """First present attribute. MCP 2.0 renamed fields to snake_case (`is_error`,
    `input_schema`); 1.x used camelCase. Reading both keeps us honest across a
    version bump rather than silently seeing None — which for `is_error` would mean
    treating every failure as a success."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _result_text(result: Any, qualified: str) -> str:
    """A CallToolResult as text, raising on the two ways MCP reports failure."""
    if _attr(result, "is_error", "isError", default=False):
        raise McpError(f"{qualified}: {_content_text(result) or 'tool reported an error'}")
    return _content_text(result)


def _content_text(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            # Non-text content (images, embedded resources) — keep the type so the
            # model knows something came back rather than seeing an empty string.
            parts.append(f"[{getattr(block, 'type', 'content')}]")
    return "\n".join(parts)


@dataclass
class Connection:
    """One connected MCP server."""

    spec: McpServer
    session: ClientSession
    available: dict[str, Any] = field(default_factory=dict)  # tool name -> MCP Tool

    async def discover(self) -> None:
        listed = await self.session.list_tools()
        self.available = {t.name: t for t in listed.tools}

        declared = {t.tool for t in self.spec.tools}
        missing = sorted(declared - self.available.keys())
        if missing:
            raise QuarantineError(
                f"mcp server {self.spec.name!r} does not expose declared tool(s): "
                f"{', '.join(missing)}")

        extra = len(self.available) - len(declared)
        if extra > 0:
            # Not an error: being strict here would break every agent whenever an
            # upstream server ships a release. They're simply never shown to the model.
            log.info("mcp %s: %d tools available, %d declared (%d ignored)",
                     self.spec.name, len(self.available), len(declared), extra)

    async def call(self, tool: str, args: dict | None) -> str:
        qualified = self.spec.qualified(tool)
        try:
            result = await self.session.call_tool(tool, args or {})
        except Exception as e:  # transport/protocol failure, as opposed to isError
            raise McpError(f"{qualified}: {type(e).__name__}: {e}") from e
        return _result_text(result, qualified)

    def properties(self, tool: str) -> dict:
        """The tool's argument schema in the shape BoundFlow's Tool wants — its
        `input_schema` is the properties map, which the SDK wraps as
        {"type": "object", "properties": ...}.

        `required` has nowhere to go in that shape, so it's folded into each
        argument's description instead of being dropped.
        """
        schema = _attr(self.available[tool], "input_schema", "inputSchema", default={})
        props = dict(schema.get("properties") or {})
        for name in schema.get("required") or []:
            if name in props:
                prop = dict(props[name])
                desc = prop.get("description", "")
                prop["description"] = f"{desc} (required)".strip()
                props[name] = prop
        return props

    def description(self, tool: str) -> str:
        return getattr(self.available[tool], "description", None) or tool


class ToolSet:
    """Every MCP server one agent version needs, connected.

    Splits the declared tools two ways, which is where `approval` stops being a
    config value and becomes structure:

    - `inline` are handed to the model as BoundFlow tools and called inside the loop.
    - `gated` are NOT — the model never receives them and can only name one in a
      proposal. They're invoked from `execute_act`, after a human approves.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._connections: dict[str, Connection] = {}

    async def connect(self, cfg: AgentConfig) -> None:
        """Connect every server the config declares. Raises QuarantineError if any
        server won't start or is missing a declared tool."""
        for spec in cfg.mcp:
            try:
                session = await self._open(spec)
            except QuarantineError:
                raise
            except Exception as e:  # noqa: BLE001
                raise QuarantineError(
                    f"mcp server {spec.name!r} failed to connect: {type(e).__name__}: {e}"
                ) from e
            conn = Connection(spec, session)
            await conn.discover()
            self._connections[spec.name] = conn

    async def _open(self, spec: McpServer) -> ClientSession:
        if spec.command:
            missing = [n for n in spec.env if n not in os.environ]
            if missing:
                raise QuarantineError(
                    f"mcp server {spec.name!r} needs {', '.join(missing)} in the "
                    f"worker's environment")
            params = StdioServerParameters(
                command=spec.command,
                args=list(spec.args),
                # Only the declared variables are passed through — a server gets the
                # credentials it was given, not the worker's whole environment.
                env={n: os.environ[n] for n in spec.env} or None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        else:
            from mcp.client.streamable_http import streamablehttp_client

            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(spec.url))

        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def aclose(self) -> None:
        await self._stack.aclose()

    async def __aenter__(self) -> ToolSet:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # ── What the worker builds an agent from ────────────────────────────────

    def inline_tools(self, cfg: AgentConfig) -> list[Tool]:
        """BoundFlow tools for everything the model is allowed to call itself."""
        tools: list[Tool] = []
        for spec in cfg.mcp:
            conn = self._connections[spec.name]
            for declared in spec.tools:
                if declared.gated:
                    continue  # proposal-only; never handed to the model
                tools.append(self._as_tool(conn, declared.tool))
        return tools

    def _as_tool(self, conn: Connection, tool: str) -> Tool:
        qualified = conn.spec.qualified(tool)

        async def handler(args: dict) -> str:
            return await conn.call(tool, args)

        return Tool(
            name=qualified,
            description=conn.description(tool),
            handler=handler,
            input_schema=conn.properties(tool),
        )

    async def call_gated(self, qualified: str, args: dict | None) -> str:
        """Invoke an approval-gated tool. Called from `execute_act` only, after a
        human has approved — never from inside the agent loop."""
        server, tool = split_qualified(qualified)
        conn = self._connections.get(server)
        if conn is None:
            raise McpError(f"{qualified}: no such server")
        declared = next((t for t in conn.spec.tools if t.tool == tool), None)
        if declared is None:
            raise McpError(f"{qualified}: not declared by this agent")
        if not declared.gated:
            raise McpError(f"{qualified}: not an approval-gated tool")
        return await conn.call(tool, args)
