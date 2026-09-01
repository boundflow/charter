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

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..config.agent import AgentConfig, McpServer

log = logging.getLogger(__name__)


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
    headers = {name: _expand(value) for name, value in spec.headers.items()}
    return {"transport": "streamable_http", "url": spec.url,
            **({"headers": headers} if headers else {})}


_VAR = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _expand(value: str) -> str:
    """Fill ${VAR} from the environment, leaving anything unset as it stands.

    Left rather than blanked on purpose: a header that arrives as
    `Bearer ${GITHUB_TOKEN}` gets a 401 naming the header, which is a better
    failure than `Bearer ` and a server wondering what you meant.
    """
    return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


def _leaves(e: BaseException) -> list[str]:
    """Every real exception inside a group, flattened.

    The stdio transport fails inside a TaskGroup, and `str()` on one of those is
    "unhandled errors in a TaskGroup (1 sub-exception)" no matter what actually
    went wrong. The causes are in `.exceptions`, sometimes nested, so this walks
    down to the leaves. Duck-typed rather than `isinstance(e, BaseExceptionGroup)`
    so it still works on 3.10, which has no such builtin.
    """
    inner = getattr(e, "exceptions", None)
    if not inner:
        return [f"{type(e).__name__}: {e}"]
    return [leaf for sub in inner for leaf in _leaves(sub)]


@dataclass
class Startup:
    """Why a declared server didn't come up, in a shape an operator can act on.

    Three parts, deliberately: a `code` that is stable enough to search for and
    to branch on, the process's own words kept `verbatim`, and a `hint` naming
    the next thing to try.

    The codes are decided by *how* it failed rather than by what it printed —
    exec refused, nothing spoke in time, it exited. Reading stderr to guess at a
    category is the kind of string-matching that goes quietly wrong on the day a
    dependency changes its wording, and the raw text is carried anyway, so
    guessing buys nothing.
    """

    code: str
    verbatim: str
    hint: str


def _ran(spec: McpServer) -> str:
    return " ".join([spec.command, *spec.args])


async def _startup(spec: McpServer, seconds: float = 5.0) -> Startup:
    """Run a stdio server on its own and find out what stopped it.

    The transport never surfaces this. It reports `Connection closed`, because
    from its side that is all that happened — the process exited before speaking
    the protocol. Whatever explains it went to the subprocess's stderr, which the
    adapter hands to the worker's own stderr and nobody keeps.

    So on the failure path only, start the thing ourselves and read it.
    """
    env = {**os.environ, **{n: os.environ[n] for n in spec.env if n in os.environ}}
    try:
        proc = await asyncio.create_subprocess_exec(
            spec.command, *spec.args,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env)
    except FileNotFoundError:
        return Startup(
            "command_not_found", f"no {spec.command!r} on PATH",
            f"the worker runs `{_ran(spec)}` from its own working directory, with "
            f"its own PATH — check both are what you expect")
    except Exception as e:  # noqa: BLE001 — diagnosing must not raise
        return Startup("command_not_runnable", f"{type(e).__name__}: {e}",
                       f"the worker could not execute `{spec.command}`")

    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=seconds)
    except TimeoutError:
        proc.kill()
        return Startup(
            "no_handshake", f"still running after {seconds:g}s, having said nothing",
            f"`{_ran(spec)}` started but never spoke MCP — check it is an MCP "
            f"server and that it uses stdio rather than http")
    finally:
        if proc.returncode is None:
            proc.kill()

    text = (err or b"").decode("utf-8", "replace").strip()
    if not text:
        return Startup(
            "exited_silently", f"exited with code {proc.returncode}, saying nothing",
            f"run `{_ran(spec)}` yourself — it fails the same way outside Charter")
    # Its last line names the failure; everything above is traceback.
    return Startup(
        "startup_failed", text.splitlines()[-1].strip(),
        f"run `{_ran(spec)}` yourself to see all of it. A server that imports "
        f"fine for you and not for the worker is usually a different interpreter")


async def _why_it_failed(spec: McpServer, e: Exception) -> str:
    """The most useful thing available about why a server didn't connect.

    An operator should not have to go and find the worker's console to learn that
    a package is missing.
    """
    transport = "; ".join(_leaves(e)) or str(e)
    if not spec.command:
        return (f"mcp server {spec.name!r} is unreachable [unreachable]\n"
                f"  it said: {transport}\n"
                f"  url:     {spec.url}\n"
                f"  try:     check the server is up and the url is reachable "
                f"from the worker")

    found = await _startup(spec)
    return (f"mcp server {spec.name!r} would not start [{found.code}]\n"
            f"  it said: {found.verbatim}\n"
            f"  running: {_ran(spec)}\n"
            f"  try:     {found.hint}\n"
            f"  (transport reported: {transport})")


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
        # NOTE: `handle_tool_errors` is left at its default of True, and that
        # currently costs us failure counting. The adapter turns a failing tool
        # into a *returned* error string, and BoundFlow's wrapper counts a failure
        # only when a call raises — so an MCP tool failure is recorded as a
        # success. tool_failure_counts stays empty, max_tool_failures never trips,
        # and `on_failure: fail` never fires. Same shape as reading `isError` under
        # its 1.x name, which this file has already been bitten by once.
        #
        # Setting it False is worse, not better: the exception then propagates out
        # of the whole invocation, so one failing tool kills the run instead of
        # being reported to the model, and the agent loses the chance to work
        # around it. Measured, not assumed.
        #
        # The real fix is to classify the returned error in the governed wrapper,
        # the way a policy denial already is — count it *and* let the model read
        # it. Tracked separately rather than bodged here.
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
                raise QuarantineError(await _why_it_failed(spec, e)) from e
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

    def with_timeout(self, seconds: float) -> "ToolSet":
        """Bound how long any one tool call may take.

        Applied here rather than through the adapter's own setting because that one
        only exists for HTTP connections — a stdio server has no timeout at all, so
        a hung one blocks until the control plane cancels the entire round. Wrapping
        the tool covers both transports the same way, and turns a hang into a
        failure the agent can read.
        """
        self._tool_seconds = seconds
        return self

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
                out.append(_bounded(tool, getattr(self, "_tool_seconds", 0.0)))
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


def _bounded(tool, seconds: float):
    """The tool, with a ceiling on how long one call may take."""
    if not seconds:
        return tool

    inner = tool.coroutine or tool.func
    name = tool.name
    # MCP tools are built with `content_and_artifact`, so a bare string is rejected
    # before anything sees it. Whatever the tool promises to return, the timeout has
    # to return the same shape.
    two_part = getattr(tool, "response_format", None) == "content_and_artifact"

    async def run(*args, **kwargs):
        # Cancelled by hand rather than by `wait_for`, because the adapter opens a
        # session per call: cancelling mid-read tears the stdio transport down and
        # it raises BrokenResourceError while unwinding, which replaced the
        # TimeoutError below and took the timeout back out of the agent's hands.
        # Once the deadline has passed the outcome is decided, so whatever the
        # transport says on the way out is not a new fact.
        task = asyncio.ensure_future(inner(*args, **kwargs))
        done, _ = await asyncio.wait({task}, timeout=seconds)
        try:
            if done:
                return task.result()
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 — see above
                pass
            raise TimeoutError
        except TimeoutError:
            # Returned, not raised. A hung tool is a failed tool, and the config
            # already says what to do about one — `on_failure` and
            # `max_tool_failures` decide. Raising took that decision away: it ended
            # the agent's run wherever it stood, so `on_failure: continue` was
            # honoured when the tool errored and ignored when it hung. The wording
            # matches what a tool that fails without raising returns, so it is
            # counted and classified down the same path as any other.
            failed = (f"Error executing tool {name}: no response within "
                      f"{seconds:g}s")
            return (failed, None) if two_part else failed

    tool.coroutine = run
    return tool
