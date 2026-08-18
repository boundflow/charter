"""Against a real MCP server process — real transport, real discovery, real errors.

Everything here is also covered by fakes in test_mcp.py. It's duplicated on purpose:
the fakes were written against MCP 1.x field names and passed happily while the
installed SDK used 2.0's, so a failing tool looked like a successful one. Only a
real server catches that.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from charter.config.agent import McpServer, ToolSpec
from charter.mcp.client import McpError, QuarantineError, ToolSet

SERVER = str(Path(__file__).parent / "mcp_fixture_server.py")


def spec(tools):
    return McpServer(name="tickets", command=sys.executable, args=[SERVER], tools=tools)


class Cfg:
    def __init__(self, server):
        self.mcp = [server]


DECLARED = [
    ToolSpec(tool="get_ticket"),
    ToolSpec(tool="close_ticket", approval="always", on_failure="fail"),
]


async def _connected(tools=None):
    cfg = Cfg(spec(tools or DECLARED))
    ts = ToolSet()
    await ts.connect(cfg)
    return cfg, ts


def run(coro):
    return asyncio.run(coro)


def test_connects_and_discovers():
    async def go():
        cfg, ts = await _connected()
        try:
            assert set(ts._connections["tickets"].available) == {
                "get_ticket", "close_ticket", "undeclared_danger"}
        finally:
            await ts.aclose()
    run(go())


def test_only_declared_ungated_tools_reach_the_model():
    async def go():
        cfg, ts = await _connected()
        try:
            names = [t.name for t in ts.inline_tools(cfg)]
            # close_ticket is gated, undeclared_danger isn't declared at all.
            assert names == ["tickets__get_ticket"]
        finally:
            await ts.aclose()
    run(go())


def test_real_schema_is_extracted():
    async def go():
        cfg, ts = await _connected()
        try:
            tool = ts.inline_tools(cfg)[0]
            assert tool.input_schema["ticket_id"]["type"] == "string"
            assert "(required)" in tool.input_schema["ticket_id"]["description"]
        finally:
            await ts.aclose()
    run(go())


def test_real_tool_call():
    async def go():
        cfg, ts = await _connected()
        try:
            tool = ts.inline_tools(cfg)[0]
            assert "4821" in await tool.handler({"ticket_id": "4821"})
        finally:
            await ts.aclose()
    run(go())


def test_real_tool_failure_raises():
    """The one that fakes missed. A raising tool comes back as a successful
    JSON-RPC response with is_error set — it must become an exception, or BoundFlow
    counts a failure as a success and on_failure/tool_failures stop working."""
    async def go():
        cfg, ts = await _connected()
        try:
            with pytest.raises(McpError, match="already closed"):
                await ts.call_gated("tickets__close_ticket", {"ticket_id": "4821"})
        finally:
            await ts.aclose()
    run(go())


def test_missing_declared_tool_quarantines():
    async def go():
        ts = ToolSet()
        with pytest.raises(QuarantineError, match="does not expose declared tool"):
            await ts.connect(Cfg(spec([ToolSpec(tool="not_a_real_tool")])))
        await ts.aclose()
    run(go())
