# The leads pipeline

Find people worth talking to, connect with each of them, get a human to sign off
the first message, and keep the conversation — with the waiting, the approvals and
the durability that implies. No LinkedIn: `network.py` is a fake professional
network with SQLite behind it, so the whole thing runs locally with nothing at
stake and nobody real on the other end.

It is one agent and a config file. There is no pipeline code.

## The files

    worker.yaml                          the deployment — which control plane, whose
                                         credentials, which agents this process serves

    leads-finder/
      v1.yaml                            behaviour at v1: objective, tools, answer
                                         shape. immutable — you write v2, you don't
                                         edit v1
      v1/skills/boundflow/SKILL.md       what it knows. versioned with v1, because a
                                         rollback has to restore what the agent knew
      runtime.yaml                       policy: spend ceilings, what it may reach,
                                         how long a human has. mutable, re-applied
      lifecycle.yaml                     what the control plane does to the agent
                                         between tasks — pause it, cool it down,
                                         roll it back

    network.py                           the fake network, as an MCP server
    approve.py                           you, approving what the agent wants to send
    inbox.py                             you, as the people being contacted

Every field the schema accepts appears in those four YAML files. Anything the demo
doesn't use is commented rather than omitted, so the file is the reference.

**Which half goes where.** `charter push` seals `v1.yaml` + `v1/skills/` into an
artifact; `charter apply` sends `runtime.yaml` and `lifecycle.yaml` to the control
plane. Behaviour is packaged because it has to reach workers that may not exist
yet. Policy is applied because it has to stay changeable — a budget you cannot
lower without cutting a release is not a budget.

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

    charter agent create leads-finder --path demo/leads   # prints an instance id
    charter apply demo/leads/worker.yaml --all

    cd demo/leads && charter worker .

Both gated tools stop for a human, and nothing here routes a notification, so run
the approval console in a second terminal or the agent sits at its first gate
until it times out:

    python demo/leads/approve.py            # read each one, decide
    python demo/leads/approve.py --auto     # approve everything, hands-off

And the inbox in a third, which is you as the people being contacted:

    python demo/leads/inbox.py

Then, from the repo root, naming the instance `create` printed:

    charter run leads-finder --path demo/leads --instance <id> --topic agent-governance

Every command that acts on an agent names an instance, including when there is
only one — `charter agents` lists them if you lose the id.

The conversation is in `network.db` — delete it to start over.

## Timing

Nothing here is on a clock. The agent waits, and someone accepts or replies when
you do it in `inbox.py`. If you want to see it give up on someone, just never
accept them.

How long it sleeps between checks is its own choice, bounded by
`max_wait_seconds` in `runtime.yaml` — 5 minutes here, because a demo you cannot
watch is not a demo. Left at a production ceiling the agent will happily pick two
hours, which is correct behaviour and useless to sit through. That limit is policy
rather than versioned behaviour, so `charter apply` changes it on the next round
without restarting the worker.

The agent doesn't know any of that. It calls `connection_status` and
`conversation` and gets whatever is true, which is exactly what it would do
against a real network.
