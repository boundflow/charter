# The leads pipeline

Find people worth talking to, connect with each of them, get a human to sign off
the first message, and keep the conversation — with the waiting, the approvals and
the durability that implies. No LinkedIn: `network.py` is a fake professional
network with SQLite behind it, so the whole thing runs locally with nothing at
stake and nobody real on the other end.

It is two agents and a config file. There is no pipeline code.

## What it exercises

    leads-finder                        outreach (one instance per person)
    ─────────────                       ──────────────────────────────────
    search_people                       send_connection_request   ← human signs off
    start_async_task  ×N  ──────────▶   wait ... connection_status
    wait                                send_message              ← human signs off
    check_async_task                    wait ... conversation
    submit_result                       submit_result

Every lead gets its own instance: its own budget, its own lifecycle policy, its
own audit trail. They run at the same time, they outlive the run that started
them, and each one parks between checks rather than holding a worker open.

Not everyone accepts. `dana` never will, so there is always a branch where an
agent waits, gives up, and reports that as a normal outcome rather than a failure.

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
    charter agent create outreach     --path demo/leads
    charter apply demo/leads/worker.yaml --all

    cd demo/leads && charter worker .

Both gated tools stop for a human, and nothing here routes a notification, so run
the console in another terminal or the agents will sit at their gates until they
time out:

    python demo/leads/approve.py            # read each one, decide
    python demo/leads/approve.py --auto     # approve everything, hands-off

Then, from the repo root:

    charter run leads-finder --path demo/leads --topic agent-governance

Watch it: `charter agents` shows an outreach instance appear per lead. The
conversation is in `network.db` — delete it to start over.

## Timing

`network.py` accepts connection requests 20s after they're sent and replies 15s
after a message, so a run takes a couple of minutes rather than a couple of days.
Both are env vars (`NETWORK_ACCEPT_SECONDS`, `NETWORK_REPLY_SECONDS`).

The agents know this, because `outreach`'s objective tells them people here move
fast. That is not a fudge — it is the config carrying a fact about the world the
agent is in. Told nothing, the same agents waited the full 30 minutes their policy
allowed, which is correct for a real network and unwatchable for a demo.
