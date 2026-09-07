"""A toy support desk, so the examples run without a Zendesk or a Stripe account.

Two agents share it: `ticket-summarizer` reads, `refund-triage` reads and refunds.
One small domain across both, so what differs between the examples is the
capability being shown rather than the scenario.

    python examples/desk.py     # start it yourself, in its own terminal

It serves MCP over HTTP on localhost:8931, which is how a real MCP server is
usually reached. The agents name it by URL, so nothing here depends on which
interpreter the worker happens to run.

State is in memory. Each worker gets a fresh desk, which is what you want from an
example: the same four tickets every run, and a refund that is gone when you
restart.
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

READ_ONLY = ToolAnnotations(readOnlyHint=True)
MUTATES = ToolAnnotations(readOnlyHint=False, destructiveHint=False)

mcp = FastMCP("desk", host="127.0.0.1", port=8931)

TICKETS = {
    "T-1041": {
        "subject": "Charged twice for one order",
        "body": "My card was charged twice on the 3rd. Same amount, same order.",
        "charge_id": "ch_88213",
        "opened": "2 days ago",
    },
    "T-1042": {
        "subject": "Package never arrived",
        "body": "Tracking says delivered but nothing came. Neighbours checked too.",
        "charge_id": "ch_88410",
        "opened": "6 hours ago",
    },
    "T-1043": {
        "subject": "Wrong size, want to exchange",
        "body": "Ordered a medium, received a small. Happy to swap, not refund.",
        "charge_id": "ch_88455",
        "opened": "1 day ago",
    },
    "T-1044": {
        "subject": "Please cancel my subscription",
        "body": "Cancel from next month. No refund needed for this one.",
        "charge_id": "ch_88501",
        "opened": "4 days ago",
    },
}

CHARGES = {
    "ch_88213": {"amount_usd": 48.00, "description": "Order #4417", "refunded_usd": 0.0},
    "ch_88410": {"amount_usd": 132.50, "description": "Order #4420", "refunded_usd": 0.0},
    "ch_88455": {"amount_usd": 61.00, "description": "Order #4425", "refunded_usd": 0.0},
    "ch_88501": {"amount_usd": 19.00, "description": "Subscription, March", "refunded_usd": 0.0},
}


@mcp.tool(annotations=READ_ONLY)
def search_tickets() -> list[dict]:
    """Every open ticket, newest concern first."""
    return [{"ticket_id": k, **v} for k, v in TICKETS.items()]


@mcp.tool(annotations=READ_ONLY)
def get_ticket(ticket_id: str) -> dict:
    """One ticket, including the charge behind it."""
    if ticket_id not in TICKETS:
        raise ValueError(f"no ticket {ticket_id}. Try one of: {', '.join(TICKETS)}")
    return {"ticket_id": ticket_id, **TICKETS[ticket_id]}


@mcp.tool(annotations=READ_ONLY)
def get_charge(charge_id: str) -> dict:
    """What was charged, and what has already been refunded against it."""
    if charge_id not in CHARGES:
        raise ValueError(f"no charge {charge_id}")
    return {"charge_id": charge_id, **CHARGES[charge_id]}


@mcp.tool(annotations=MUTATES)
def create_refund(charge_id: str, amount_usd: float) -> dict:
    """Refund against a charge. Gated: a person approves this before it runs."""
    if charge_id not in CHARGES:
        raise ValueError(f"no charge {charge_id}")
    charge = CHARGES[charge_id]
    outstanding = charge["amount_usd"] - charge["refunded_usd"]
    if amount_usd > outstanding:
        raise ValueError(
            f"{amount_usd} exceeds the {outstanding} still refundable on {charge_id}")
    charge["refunded_usd"] += amount_usd
    return {"charge_id": charge_id, "refunded_usd": amount_usd,
            "remaining_usd": charge["amount_usd"] - charge["refunded_usd"]}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
