"""A fake support-desk MCP server, for trying Charter without real credentials.

Real enough to exercise the whole path — stdio transport, tool discovery, a tool
that mutates, and one that fails on demand — with nothing at stake.

    python playground/mcp_server.py     # the worker spawns this itself

`create_refund` is the interesting one: the agent is handed it and the call is
stopped for a human, so you can watch a task park mid-flight and resume days
later on a different worker.
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

READ_ONLY = ToolAnnotations(readOnlyHint=True)
MUTATES = ToolAnnotations(readOnlyHint=False, destructiveHint=True)

mcp = FastMCP("support-desk")

TICKETS = {
    "4821": {
        "subject": "Charged twice for the same order",
        "customer": "dana@example.com",
        "body": ("I ordered once but my card shows two charges of $240 on the same "
                 "day. Please refund the duplicate."),
        "charge_ids": ["ch_9001", "ch_9002"],
        "status": "open",
    },
    "5150": {
        "subject": "Want a refund, changed my mind",
        "customer": "sam@example.com",
        "body": "I bought this three months ago and decided I don't want it.",
        "charge_ids": ["ch_7700"],
        "status": "open",
    },
}

CHARGES = {
    "ch_9001": {"amount_usd": 240, "created": "2026-08-14", "refunded": False},
    "ch_9002": {"amount_usd": 240, "created": "2026-08-14", "refunded": False},
    "ch_7700": {"amount_usd": 89, "created": "2026-05-02", "refunded": False},
}

REFUNDS: list[dict] = []


@mcp.tool(annotations=READ_ONLY)
def get_ticket(ticket_id: str) -> str:
    """Fetch a support ticket by id, with the charge ids it refers to."""
    ticket = TICKETS.get(ticket_id)
    if ticket is None:
        raise ValueError(f"no ticket {ticket_id}")
    return str(ticket)


@mcp.tool(annotations=READ_ONLY)
def list_open_tickets() -> str:
    """Every ticket still open. How a scheduled agent finds its own work."""
    return str([{"ticket_id": tid, "subject": t["subject"], "status": t["status"]}
                for tid, t in TICKETS.items() if t["status"] == "open"])


@mcp.tool(annotations=READ_ONLY)
def get_charge(charge_id: str) -> str:
    """Look up one charge."""
    charge = CHARGES.get(charge_id)
    if charge is None:
        raise ValueError(f"no charge {charge_id}")
    return str(charge)


@mcp.tool(annotations=MUTATES)
def create_refund(charge_id: str, amount_usd: float, reason: str) -> str:
    """Refund a charge. Money moves — this is the one a human approves."""
    charge = CHARGES.get(charge_id)
    if charge is None:
        raise ValueError(f"no charge {charge_id}")
    if charge["refunded"]:
        raise ValueError(f"{charge_id} was already refunded")
    charge["refunded"] = True
    REFUNDS.append({"charge_id": charge_id, "amount_usd": amount_usd, "reason": reason})
    return f"refund re_{len(REFUNDS):04d} created for {charge_id}: ${amount_usd}"


@mcp.tool()
def always_fails(why: str = "") -> str:
    """Fails every time. For watching max_tool_failures trip and name the tool."""
    raise RuntimeError(f"this tool is broken on purpose ({why or 'no reason given'})")


if __name__ == "__main__":
    mcp.run(transport="stdio")
