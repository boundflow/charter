"""A fake professional network, for running the leads pipeline without LinkedIn.

Real enough to exercise the whole thing — searching, connecting, waiting for
someone to accept, messaging once they have, and a conversation that outlives the
run that started it — with nobody real on the other end.

    python demo/leads/network.py     # the worker spawns this itself

Nobody here answers on a timer. You are the other side: `inbox.py` is where you
accept a connection request or write a reply, and until you do, the agent is
genuinely waiting on a person. That is the whole point of the pipeline, and a
canned reply after twenty seconds doesn't exercise it.

The agent's view is unchanged either way. It calls `connection_status` and
`conversation` and gets whatever is true, with no idea a human is typing the
other half.

State lives in SQLite next to this file rather than in memory, because the server,
the worker and your inbox are separate processes and a conversation is supposed to
outlive all of them. Delete network.db to start over.
"""

import sqlite3
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

READ_ONLY = ToolAnnotations(readOnlyHint=True)
MUTATES = ToolAnnotations(readOnlyHint=False, destructiveHint=False)

mcp = FastMCP("network")

DB = Path(__file__).parent / "network.db"

PEOPLE = [
    ("ade", "Ade Okonkwo", "Staff engineer, fintech",
     "agent-governance",
     "Wrote a long post about an agent that retried a payout tool 40 times "
     "overnight and nobody found out until reconciliation."),
    ("mira", "Mira Castellanos", "Founding engineer, devtools",
     "agent-governance",
     "Asked publicly how anyone puts a human approval step into a LangGraph "
     "run without rewriting the whole graph."),
    ("dana", "Dana Whitfield", "Head of platform, logistics",
     "agent-governance",
     "Complained that their agents lose everything on deploy because state "
     "lives in process memory."),
    ("tomasz", "Tomasz Nowak", "ML engineer, marketplace",
     "agent-governance",
     "Frustrated that a subagent kept running after the parent task was "
     "cancelled and there was no way to stop it."),
    ("priya", "Priya Raman", "CTO, healthtech",
     "compliance",
     "Needs an audit trail showing who approved what before agents touch "
     "patient records."),
]


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS connections (
            person_id    TEXT PRIMARY KEY,
            note         TEXT NOT NULL,
            requested_at REAL NOT NULL,
            -- NULL until you accept in inbox.py. Nothing sets this on a timer.
            accepted_at  REAL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            direction TEXT NOT NULL,   -- 'sent' or 'received'
            text      TEXT NOT NULL,
            at        REAL NOT NULL
        );
    """)
    return conn


def person(person_id: str) -> dict:
    for pid, name, headline, topic, signal in PEOPLE:
        if pid == person_id:
            return {"id": pid, "name": name, "headline": headline,
                    "topic": topic, "signal": signal}
    raise ValueError(f"no such person: {person_id}")


@mcp.tool(annotations=READ_ONLY)
def search_people(topic: str, limit: int = 5) -> str:
    """Find people whose recent activity matches a topic.

    Returns their id, name, headline, and the specific thing they said that made
    them a match — which is what an opening message should be written to.
    """
    hits = [dict(id=p[0], name=p[1], headline=p[2], signal=p[4])
            for p in PEOPLE if topic.lower() in p[3].lower()][:limit]
    if not hits:
        return f"No matches for {topic!r}. Known topics: agent-governance, compliance."
    return "\n".join(
        f"{h['id']}: {h['name']} — {h['headline']}\n    signal: {h['signal']}"
        for h in hits)


@mcp.tool(annotations=MUTATES)
def send_connection_request(person_id: str, note: str) -> str:
    """Ask to connect, with a short note. Goes to a real person — worth a human
    reading it first."""
    who = person(person_id)
    conn = db()
    with conn:
        existing = conn.execute(
            "SELECT requested_at FROM connections WHERE person_id = ?",
            (person_id,)).fetchone()
        if existing:
            return f"Already sent a request to {who['name']}."
        conn.execute(
            "INSERT INTO connections (person_id, note, requested_at) VALUES (?, ?, ?)",
            (person_id, note, time.time()))
    return (f"Connection request sent to {who['name']}. They have not accepted "
            f"yet — check back later rather than waiting on it.")


@mcp.tool(annotations=READ_ONLY)
def connection_status(person_id: str) -> str:
    """Whether someone has accepted your request yet.

    Nobody accepts instantly, and some people never do. Both are normal, and
    checking again straight away tells you nothing new.
    """
    who = person(person_id)
    conn = db()
    row = conn.execute(
        "SELECT requested_at, accepted_at FROM connections WHERE person_id = ?",
        (person_id,)).fetchone()
    if row is None:
        return f"No request has been sent to {who['name']}."
    waited = int(time.time() - row["requested_at"])
    if row["accepted_at"] is None:
        return (f"{who['name']} has not accepted yet ({waited}s since you asked). "
                f"Some people take days, and some never do.")
    return f"{who['name']} accepted your request. You can message them now."


@mcp.tool(annotations=MUTATES)
def send_message(person_id: str, text: str) -> str:
    """Message someone who has accepted. This is the one a human signs off."""
    who = person(person_id)
    conn = db()
    row = conn.execute(
        "SELECT accepted_at FROM connections WHERE person_id = ?",
        (person_id,)).fetchone()
    if row is None:
        raise ValueError(f"not connected to {who['name']} — send a request first")
    if row["accepted_at"] is None:
        raise ValueError(f"{who['name']} has not accepted yet — cannot message")
    with conn:
        conn.execute(
            "INSERT INTO messages (person_id, direction, text, at) VALUES (?,?,?,?)",
            (person_id, "sent", text, time.time()))
    return f"Message sent to {who['name']}."


@mcp.tool(annotations=READ_ONLY)
def conversation(person_id: str) -> str:
    """The whole thread with someone, oldest first.

    A reply appears when they write one, which may be a while and may be never.
    An unanswered message is not a failure and it is not a reason to send another.
    """
    who = person(person_id)
    rows = db().execute(
        "SELECT direction, text FROM messages WHERE person_id = ? ORDER BY at, id",
        (person_id,)).fetchall()
    if not rows:
        return f"No messages with {who['name']} yet."

    lines = [f"{'you' if r['direction'] == 'sent' else who['name']}: {r['text']}"
             for r in rows]
    if rows[-1]["direction"] == "sent":
        lines.append(f"({who['name']} has not replied to that yet.)")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
