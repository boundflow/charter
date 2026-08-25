# The leads pipeline

Find people worth talking to, connect with each of them, get a human to sign off
the first message, and keep the conversation — with the waiting, the approvals and
the durability that implies. No LinkedIn: `network.py` is a fake professional
network with SQLite behind it, so the whole thing runs locally with nothing at
stake and nobody real on the other end.

It is one agent and a config file. There is no pipeline code.

## What it exercises

One agent runs the whole campaign:

    search_people
    send_connection_request   ← you approve each one, with its note
    wait ... connection_status
    send_message              ← you approve each one, with its text
    wait ... conversation
    send_message (a reply)    ← you approve that too
    ... and round again while anyone might still answer

Everything that reaches a person stops for you. Everything else runs on its own.

The waiting is real: `inbox.py` is where you play the people being contacted, so
someone accepts when you accept and replies when you write a reply. Nothing moves
on a timer, which is the only way to see an agent genuinely wait on a person.

## Running it

Four environment variables — the control plane, its key, Postgres for the
harness's own state, and a model key:

    export BOUNDFLOW_SERVER_ADDRESS=http://localhost:50051
    export BOUNDFLOW_API_KEY=...
    export CHARTER_STORE_URL=postgres://...
    export ANTHROPIC_API_KEY=...
    export CHARTER_TENANT=default

Create the instances, then start the worker from this directory — the MCP server
is spawned as a subprocess relative to wherever the worker runs:

    charter agent create leads-finder --path demo/leads
    charter apply demo/leads/worker.yaml --all

    cd demo/leads && charter worker .

Both gated tools stop for a human, and nothing here routes a notification, so run
the approval console in a second terminal or the agent sits at its first gate
until it times out:

    python demo/leads/approve.py            # read each one, decide
    python demo/leads/approve.py --auto     # approve everything, hands-off

And the inbox in a third, which is you as the people being contacted:

    python demo/leads/inbox.py

Then, from the repo root:

    charter run leads-finder --path demo/leads --topic agent-governance

The conversation is in `network.db` — delete it to start over.

## Timing

Nothing here is on a clock. The agent waits, and someone accepts or replies when
you do it in `inbox.py`. If you want to see it give up on someone, just never
accept them.

The agent doesn't know any of that. It calls `connection_status` and
`conversation` and gets whatever is true, which is exactly what it would do
against a real network.
