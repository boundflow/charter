"""You, as the people the agent is reaching out to.

The agent thinks it is on a professional network. It sends a connection request
and waits; it sends a message and waits. Nothing here happens on a timer — you
decide when someone accepts and what they say back, which is the only way the
waiting is real.

    python demo/leads/inbox.py

Shows what is waiting on you and lets you answer it. Runs alongside the worker and
`approve.py`: that one is you approving your own agent's outbound messages, this
one is the other side answering. Ctrl-C to leave; nothing is lost, because it all
lives in network.db.
"""

import sqlite3
import sys
import time
from pathlib import Path

DB = Path(__file__).parent / "network.db"

PEOPLE = {
    "ade": "Ade Okonkwo",
    "mira": "Mira Castellanos",
    "dana": "Dana Whitfield",
    "tomasz": "Tomasz Nowak",
    "priya": "Priya Raman",
}


def db() -> sqlite3.Connection:
    if not DB.exists():
        sys.exit(f"no {DB.name} yet — run the agent first, it creates it")
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def waiting(conn) -> list[tuple[str, str, str]]:
    """What needs you: (person_id, kind, context).

    Two kinds. An unaccepted request is someone waiting to be let in; a trailing
    outbound message is someone who has been written to and hasn't answered.
    """
    out = []
    for row in conn.execute(
            "SELECT person_id, note, accepted_at FROM connections ORDER BY requested_at"):
        if row["accepted_at"] is None:
            out.append((row["person_id"], "request", row["note"]))
            continue
        last = conn.execute(
            "SELECT direction, text FROM messages WHERE person_id = ? "
            "ORDER BY at DESC, id DESC LIMIT 1", (row["person_id"],)).fetchone()
        if last and last["direction"] == "sent":
            out.append((row["person_id"], "message", last["text"]))
    return out


def accept(conn, person_id: str) -> None:
    with conn:
        conn.execute("UPDATE connections SET accepted_at = ? WHERE person_id = ?",
                     (time.time(), person_id))


def reply(conn, person_id: str, text: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO messages (person_id, direction, text, at) VALUES (?,?,?,?)",
            (person_id, "received", text, time.time()))


def wrap(text: str, indent: str = "    ") -> str:
    import textwrap
    return textwrap.fill(text, 76, initial_indent=indent, subsequent_indent=indent)


def main() -> None:
    conn = db()
    print("\nyou are the other side. ctrl-c to leave.\n")
    while True:
        pending = waiting(conn)
        if not pending:
            print("nothing waiting on you — leave this open, the agent is working")
            time.sleep(3)
            # Reconnect so a write from the worker's process is visible.
            conn = db()
            continue

        for person_id, kind, context in pending:
            name = PEOPLE.get(person_id, person_id)
            print(f"\n── {name}")
            if kind == "request":
                print("   wants to connect:")
                print(wrap(context))
                answer = input("\n   accept? [y/N/skip] ").strip().lower()
                if answer == "y":
                    accept(conn, person_id)
                    print("   connected")
                elif answer in ("s", "skip"):
                    # Left pending, so it comes back next time round. This is how
                    # you play someone who simply hasn't got to it yet.
                    print("   left waiting")
                else:
                    print("   ignored for now")
            else:
                print("   said to you:")
                print(wrap(context))
                answer = input("\n   your reply (blank to say nothing yet): ").strip()
                if answer:
                    reply(conn, person_id, answer)
                    print("   sent")
                else:
                    print("   left unanswered")
        conn = db()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nleft the inbox")
