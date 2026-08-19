"""A real MCP server over stdio, used by test_mcp_live.

Kept as a module so the live test spawns an actual process and exercises the real
transport, discovery, and error paths. That has caught two things fakes didn't:
Charter reading `isError` against an SDK that had renamed it `is_error`, which
recorded every tool failure as a success, and dotted tool names the provider
rejects.

Annotation names are camelCase here because that is what `ToolAnnotations` uses
on the wire, which is also how they arrive on a LangChain tool's `metadata`.
"""
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("tickets")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_ticket(ticket_id: str) -> str:
    """Fetch a support ticket by id."""
    return f"ticket {ticket_id}: customer wants a refund"


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def close_ticket(ticket_id: str) -> str:
    """Close a ticket."""
    raise ValueError(f"ticket {ticket_id} is already closed")


@mcp.tool()
def undeclared_danger(x: str) -> str:
    """A tool no agent config declares."""
    return "should never be reachable"


if __name__ == "__main__":
    mcp.run(transport="stdio")
