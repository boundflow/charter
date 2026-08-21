# Running workers

A worker is the half of Charter that runs on your side: it makes the model calls,
talks to your MCP servers, and holds the agent's filesystem. Charter's control
plane holds identity, policy, budgets, gates and the audit trail. Your model key
and your data stay here.

    cp .env.example .env      # fill in your keys and point at your project
    docker compose up

The image installs `boundflow` from PyPI, and the governor Charter runs on
(`run_governed`, `agent_governor`) is not in a release yet — so the build succeeds
and the worker exits on import until one ships. Until then, run the worker from a
checkout with the SDK installed alongside it.

That's a Postgres and one worker. For more:

    docker compose up --scale worker=3

## Why the workers share a database

Not for speed. A task that stops at an approval gate resumes later — possibly days
later, and on whichever worker happens to be running then. That only works if the
worker picking it up can read the state the one that parked it wrote. Give each
worker its own database and every task can only ever resume on the machine it
started on, which defeats the point of it being durable at all.

So "a worker pool" isn't something you declare. Workers that share `store.url` and
serve the same agents are a pool; `--scale` is the whole mechanism.

## What goes where

| your side | Charter |
| --- | --- |
| model calls, and the key for them | identity, versions, rollback |
| MCP servers and their credentials | budgets and policy |
| the agent's files and conversation | approval gates and the audit trail |
| this Postgres | scheduling and waking parked tasks |

## Changing an agent

The project is mounted read-only, not baked into the image, so editing an
objective is a restart rather than a rebuild:

    charter apply worker.yaml --all
    docker compose restart worker

The image only changes when Charter itself does.

## Before you scale past one

Every worker in a fleet should serve the same agents *and the same versions*. A
task is dispatched at the version it started on, and a worker only handles the
versions its `serves:` lists — so a worker that has dropped v1 will not pick up a
v1 task, and that task waits without complaining. When rolling out v2, keep v1 in
`serves:` until nothing is parked on it.
