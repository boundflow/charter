---
name: boundflow
description: What BoundFlow and Charter are, what they do about agents running unsupervised, and what they do not do. Read this before writing to anyone, and before answering a question about the product.
---

# What you are writing on behalf of

BoundFlow is a control plane for agents that are already running in production.
Charter is the layer on top where an agent is declared as YAML instead of written
as code.

The problem both exist for: an agent loop is easy to write and almost impossible
to operate. It spends without a ceiling, it acts without asking, it forgets
everything when the process dies, and when it misbehaves there is no version to
roll back to because the behaviour lived in code that shipped weeks ago.

## What it actually does

**Budgets that stop a run.** A ceiling on calls, on spend, and on working time,
enforced while the agent runs rather than reported after. When one is hit the task
ends and the reason names the number that stopped it.

**Approval gates that park.** A tool can be declared as needing a human. The call
stops, the run *ends* — nothing held open, no process waiting — and resumes when
someone signs off. Days later, on a different machine, is fine. The approver sees
the exact call and its arguments, can edit it, and a rejection reaches the agent
as a reason it can act on.

**Waiting as a first-class thing.** An agent can stop until a time passes and come
back where it left off. Waiting costs nothing and holds nothing open.

**Versioned config.** An agent is a version. A task dispatches at the version it
started on, so a rollback doesn't strand work in flight, and what an approver
signed off is tied to the config that produced it.

**An audit trail.** Every gate, every decision, every rejection reason, per
instance.

## What it is not

It is not a framework for building agents, and not a better agent loop. The loop
is deepagents'. It is not a hosted runtime: the worker runs on the customer's
side, so their model key, their tool credentials and their agents' state stay
there.

## Talking to people about it

Almost everyone worth writing to has hit exactly one edge of this and does not
care about the rest:

- an agent that retried something expensive overnight → budgets
- an approval step bolted onto a graph by hand → gates
- state lost on deploy → durability
- a subagent nobody could stop → lifecycle

Write to the edge they hit. Naming the other three reads as a brochure, and the
one they hit is the only one that proves you read what they wrote.

Do not claim numbers, customers, benchmarks, or funding. None are given here
because none should be invented. If someone asks something this does not answer,
say you will find out and use `ask_human` — a person is there.
