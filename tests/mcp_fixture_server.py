"""A real MCP server over stdio, used by test_mcp_live.

Kept as a module so the live test spawns an actual process and exercises the real
transport, discovery, and error paths — the fakes in test_mcp.py all passed while
Charter was reading `isError` against an SDK that had renamed it to `is_error`,
which meant every tool failure was being recorded as a success.
"""
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

mcp = MCPServer("tickets")


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def get_ticket(ticket_id: str) -> str:
    """Fetch a support ticket by id."""
    return f"ticket {ticket_id}: customer wants a refund"


@mcp.tool()
def close_ticket(ticket_id: str) -> str:
    """Close a ticket."""
    raise ValueError(f"ticket {ticket_id} is already closed")


@mcp.tool()
def undeclared_danger(x: str) -> str:
    """A tool no agent config declares."""
    return "should never be reachable"


if __name__ == "__main__":
    mcp.run()
