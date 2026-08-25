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

## Serving agents from a registry

`serves:` takes either a directory or an OCI reference:

    serves:
      - agent: leads-finder     # from the mounted project — how you develop
        versions: [1]
      - ref: ghcr.io/acme/agents/leads-finder:v1   # how you deploy

A reference is pulled at boot. The artifact names its own agent and version, so
neither is repeated here, and the worker needs no checkout at all.

Credentials are whatever `docker login` already wrote: the compose file mounts
your Docker config read-only and points `DOCKER_CONFIG` at it. Nothing is
Charter-specific, which is the point — a Docker config holds several registries at
once and carries credential helpers, which is what makes ECR and GCR work at all,
since a static password there expires within the day.

On Kubernetes the same secret serves twice — as the `imagePullSecret` the kubelet
uses for this image, and mounted as a file for the worker to read:

    volumes:
      - name: registry
        secret:
          secretName: ghcr-creds
          items: [{key: .dockerconfigjson, path: config.json}]
    env:
      - name: DOCKER_CONFIG
        value: /etc/registry

They are not interchangeable by default: `imagePullSecrets` is for the kubelet
pulling the worker image, and the worker pulling an agent artifact is an ordinary
HTTPS request from inside the process. It needs its own copy.

## Tools the artifact does not carry

An agent that declares a stdio MCP server — `command: python, args: [thing.py]` —
depends on that process existing on the worker. The artifact carries the config,
not the tool, and `charter push` says so when you publish one.

Two ways to satisfy it:

    FROM ghcr.io/boundflow/charter-worker:1.2
    COPY thing.py /opt/mcp/            # the tool ships in your worker image

or run the server as its own container and declare it by url, which is what a
sidecar is:

    mcp:
      - name: net
        url: http://localhost:8080/mcp

Loopback is allowed for exactly this reason — a sidecar shares the pod's network
namespace, so the traffic never leaves it. Anything else must be https.

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
