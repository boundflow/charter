"""MCP servers, loaded by the ecosystem adapter and bounded by our config.

Charter used to speak MCP itself — open a session, `tools/list`, dispatch calls,
unwrap content blocks. All of that is `langchain-mcp-adapters`' job now, and it
produces exactly the LangChain tools the harness and `ctx.agent_tools()` want.
Writing it ourselves cost us two bugs a fake couldn't catch: dotted tool names
that Anthropic rejects, and reading `isError` from an SDK that had renamed it
`is_error`, which recorded every tool failure as a success.

Two things stay ours, because nothing upstream does them:

**The declaration.** A version names the servers and the tools it uses. Tools the
server offers but the config didn't declare are never handed to the model, and a
declared tool the server doesn't expose quarantines the agent rather than failing
mid-task.

**The ratchet.** `ToolAnnotations` are hints, and the spec is explicit that a
client shouldn't make trust decisions from them on an untrusted server. But the
risk is one-directional: a server newly marking something destructive should gate
it immediately, while a server marking something read-only must not *remove* a
gate — that's a decision belonging in a file with an author. So annotations may
only ever tighten.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from ..config.agent import AgentConfig, McpServer

log = logging.getLogger(__name__)


class McpError(Exception):
    """A tool call failed — the server said so, or the transport did."""


class QuarantineError(Exception):
    """This agent version can't be served: a server is unreachable, or doesn't
    expose something the config declared."""


def approval_for(annotations: Any) -> tuple[str, str]:
    """What a tool's annotations suggest, and why, in one word each.

    The defaults fall the right way round in the spec: `read_only_hint` defaults
    to false, and `destructive_hint` defaults to **true** when a tool isn't
    read-only. A server that says nothing is treated as dangerous.

    Accepts the adapter's metadata dict (camelCase, as MCP puts it on the wire) or
    anything with the snake_case attributes, since `charter import` and the loader
    reach this from different directions.
    """
    def hint(camel: str, snake: str) -> Any:
        if isinstance(annotations, dict):
            return annotations.get(camel)
        return getattr(annotations, snake, None)

    # The reason is the annotation's own name, so "why is this gated" is greppable
    # against the MCP spec rather than against a word we invented.
    if annotations is None:
        return "always", "unannotated"
    if hint("readOnlyHint", "read_only_hint"):
        return "never", "read_only_hint"
    if hint("destructiveHint", "destructive_hint") is False:
        return "always", "not_destructive_hint"
    return "always", "destructive_hint"


def _connection(spec: McpServer) -> dict:
    """One server in the adapter's connection vocabulary.

    `env` names variables to pass through rather than carrying values, so a config
    file never holds a credential. A name that isn't set is dropped rather than
    passed as empty, which fails at the server with a clearer message than an
    empty key would.
    """
    if spec.command:
        env = {name: os.environ[name] for name in spec.env if name in os.environ}
        return {"transport": "stdio", "command": spec.command,
                "args": list(spec.args), **({"env": env} if env else {})}
    return {"transport": "streamable_http", "url": spec.url}


@dataclass
class Server:
    """One declared server, once its tools are loaded."""

    spec: McpServer
    tools: dict[str, Any] = field(default_factory=dict)   # bare name -> LangChain tool
    # Tools the server's annotations gated beyond what the config asked for.
    tightened: dict[str, str] = field(default_factory=dict)

    def gated(self, tool: str) -> bool:
        declared = next(t for t in self.spec.tools if t.tool == tool)
        return declared.gated or tool in self.tightened


class ToolSet:
    """Every MCP tool one agent version may use, loaded once at boot.

    The adapter manages sessions per call, so there's no connection to hold open
    or lose — at the cost of a stdio server being spawned per call. Worth knowing
    before pointing an agent at a slow-starting local server; an HTTP server has
    no such cost.
    """

    def __init__(self) -> None:
        self.servers: dict[str, Server] = {}
        self._client: Any = None

    async def connect(self, cfg: AgentConfig) -> None:
        """Load every declared server's tools, or quarantine the agent."""
        if not cfg.mcp:
            return
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # Never `tool_name_prefix=True`: the adapter joins with a single underscore,
        # so a server named `desk_get` with a tool `ticket` collides with `desk` and
        # `get_ticket`. Charter's `__` is unambiguous, and it's already baked into
        # tool_call_limits and lifecycle rules.
        self._client = MultiServerMCPClient(
            {s.name: _connection(s) for s in cfg.mcp})

        # Asked per server rather than all at once: get_tools() flattens every
        # server's tools into one list of bare names, and we need to know which
        # server offered what to namespace it and to check the declaration.
        by_server = await self._by_server(cfg)

        for spec in cfg.mcp:
            available = by_server[spec.name]
            declared = {t.tool for t in spec.tools}
            if missing := sorted(declared - available.keys()):
                raise QuarantineError(
                    f"mcp server {spec.name!r} does not expose declared tool(s): "
                    f"{', '.join(missing)}")

            if extra := len(available) - len(declared):
                # Not an error: being strict would break every agent whenever an
                # upstream server ships a release. They're simply never declared,
                # so the model never sees them.
                log.info("mcp %s: %d available, %d declared (%d ignored)",
                         spec.name, len(available), len(declared), extra)

            server = Server(spec=spec, tools={n: t for n, t in available.items()
                                              if n in declared})
            self._ratchet(server, available)
            self.servers[spec.name] = server

    async def _by_server(self, cfg: AgentConfig) -> dict[str, dict[str, Any]]:
        """Tools grouped by the server that offers them."""
        out: dict[str, dict[str, Any]] = {}
        for spec in cfg.mcp:
            try:
                tools = await self._client.get_tools(server_name=spec.name)
            except Exception as e:
                raise QuarantineError(f"mcp server {spec.name!r}: {e}") from e
            out[spec.name] = {t.name: t for t in tools}
        return out

    def _ratchet(self, server: Server, available: dict[str, Any]) -> None:
        """Resolve each declared tool's approval, honouring the server's hints —
        upward only. Explicit config always wins; a server can make an agent safer
        without a deploy, but never less safe."""
        rules = server.spec.approval
        for declared in server.spec.tools:
            if declared.approval is not None or rules is None:
                continue
            tool = available.get(declared.tool)
            suggested, why = approval_for(getattr(tool, "metadata", None))
            wanted = rules.read_only if suggested == "never" else rules.default
            if wanted == "always":
                server.tightened[declared.tool] = why
                log.info("mcp %s: %s gated by policy (%s)",
                         server.spec.name, declared.tool, why)

    def langchain_tools(self) -> list:
        """Every declared tool, renamed to `server__tool` and ready for the harness.

        Renaming rather than wrapping: the harness, the governor's per-tool caps,
        and lifecycle rules all key on the name the model sees, so there must be
        exactly one spelling of it.
        """
        out = []
        for server in self.servers.values():
            for name, tool in server.tools.items():
                tool.name = server.spec.qualified(name)
                out.append(tool)
        return out

    def gated_tools(self) -> list[str]:
        """Declared tools whose call stops for a human, namespaced.

        Resolved from config *and* the ratchet, so it includes tools a server's
        annotations gated after the config was written."""
        return [s.spec.qualified(t) for s in self.servers.values()
                for t in s.tools if s.gated(t)]

    async def aclose(self) -> None:
        self._client = None

    async def __aenter__(self) -> ToolSet:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


async def probe(*, command: str = "", args: list[str] | None = None,
                url: str = "", env: list[str] | None = None) -> list:
    """Every tool a server offers, for `charter import`. No config needed — this
    runs before there is one."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if command:
        passthrough = {n: os.environ[n] for n in (env or []) if n in os.environ}
        conn = {"transport": "stdio", "command": command, "args": list(args or []),
                **({"env": passthrough} if passthrough else {})}
    else:
        conn = {"transport": "streamable_http", "url": url}

    try:
        return await MultiServerMCPClient({"probe": conn}).get_tools()
    except Exception as e:
        raise QuarantineError(f"mcp: {e}") from e
