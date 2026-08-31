# Charter

**The easiest way to build and manage production-ready agents.**

> **Pre-alpha, and in the open early.** The design is settled enough to read and
> argue with; the code is not settled enough to run anything you care about.
> Charter currently needs an unreleased branch of
> [BoundFlow](https://github.com/boundflow/boundflow) — see
> [Getting started](#getting-started).

A prototype agent is a prompt and some tools. A production agent needs more: a
defined responsibility, explicit authority over what it may touch, a budget it
can't exceed, a way for humans to intervene while it works, and someone watching
its behavior over time.

Charter is that, as configuration. You describe an agent in YAML — what it's
responsible for, which tools it gets, what needs a human, what it may spend — and
Charter deploys it as a durable, governed service you can operate.

The agent runs in your environment. Its state, policy and history live in a
control plane, so a task survives a closed laptop, a restarted worker, or an
approval that takes until tomorrow.

> [!WARNING]
> **Pre-alpha.** Charter agents have run end-to-end against a live control plane,
> a real model and real MCP servers — but plenty hasn't. Expect rough edges and
> expect the configuration format to change.

## Define an agent

An agent is a directory with a version file in it — `refund-triage/v1.yaml`:

```yaml
apiVersion: charter/v1
kind: AgentConfig

name: refund-triage
version: 1
model: claude-haiku-4-5

objective: |
  Resolve the refund request on ticket {{ inputs.ticket_id }}.
  Look up the ticket and the charge before proposing anything.
  Never propose a refund above ${{ inputs.max_refund_usd }}.

inputs:
  ticket_id: { type: string, required: true }
  max_refund_usd: { type: number, default: 100 }

mcp:
  - name: stripe
    url: https://mcp.stripe.com
    env: [STRIPE_API_KEY]
    tools:
      - tool: get_charge
      - tool: create_refund
        approval: always

outcome:
  deliverable:
    resolution: { type: string }
    refunded_usd: { type: number }
```

That file is the whole agent. Budgets and lifecycle rules live in two optional
files beside it — see [a project](#a-project) — and a conservative default
ceiling applies until you add them.

Deploy it, create an instance, and give it work:

```bash
charter agent create refund-triage
charter apply .
charter run refund-triage --instance a3f9c012 --ticket-id 4821
```

Suppose the agent reads the ticket and the charge and concludes a $240 refund is
warranted. It cannot issue it. `create_refund` requires approval, so Charter
parks the task and surfaces the proposed action and the agent's reasoning to a
human:

```bash
charter approve apr_01J8Z --reason "third dispute this month"
```

Only then is the refund executed. The agent receives the result, finishes the
task, and reports what happened. The task didn't live in your terminal in the
meantime.

## What makes it production-ready

### Explicit authority

Access to an MCP server is not access to everything that server exposes. The
model only ever sees the tools you declared:

```
mcp stripe: 34 tools available, 2 declared (32 ignored)
```

If the server adds a tool tomorrow, your agent doesn't silently gain a capability.
And tools marked `approval: always` aren't callable inside the agent loop at all —
the agent can propose them, but a separate step executes them after a human signs
off. The model can propose authority it doesn't have; it can't grant it to itself.

### A budget per task

```yaml
per_run:
  max_cost_usd: 0.30
  max_llm_calls: 40
  max_tool_failures: 3
```

Limits hold across the whole task — reasoning rounds, retries, human feedback and
tool calls alike. When one is exhausted, Charter tells you the operational reason
rather than collapsing it into a generic failure:

```
stripe__create_refund failed 3 times (max_tool_failures=3)
the integration looks broken
```

### Humans in the loop, durably

An agent stops for a human for two reasons: it needs information it shouldn't
guess, or it wants to do something beyond its authority.

```bash
charter pending refund-triage
charter approve apr_01J8Z --reason "confirmed duplicate"
charter answer  inp_01J8Z "use the March charge"
```

Neither is a process sitting around waiting. Charter checkpoints the task; it can
wait overnight and resume on another worker with everything it had discovered,
spent and been told. Rejection carries a reason back to the agent, so a "no" is
feedback it can act on, not just a closed door.

### Policy that acts on its own

Humans govern individual actions. Lifecycle rules govern the agent itself:

```yaml
rules:
  - when: { metric: num_failures, threshold: 2 }
    then: { pause: { window: 5 } }

  - when: { metric: cost, threshold: 5.00 }
    then: { cooldown: { window: 20, seconds: 300 } }

  - when: { metric: approval_rejections, threshold: 3 }
    then: { set_version: { target: 1 } }
```

A noisy agent gets cooled down. A repeatedly failing one gets paused. A new
version whose decisions keep getting rejected rolls back to the one that worked —
and because versions are immutable specifications, rollback restores the whole
agent, not just a prompt string. What changed is a file in Git, with an author and
a diff.

## Operate agents, not sessions

An agent persists beyond any single task.

```
$ charter agents

AGENT           VER  STATUS  ACTIVITY
invoice-chaser  v1   paused  idle
refund-triage   v1   active  awaiting_approval
ticket-sweeper  v1   active  idle
```

Inspect its authority and how close it is to tripping a rule:

```
$ charter describe refund-triage

limits per task
  max_cost_usd    0.25
  max_llm_calls   20

rules
  num_failures   1 of 2     -> pause window=5
  cost           5.2 of 5   -> cooldown window=20 seconds=300
```

And reconstruct, afterwards, every governance decision that was made:

```
$ charter audit refund-triage

2026-08-18 03:47  approval rejected by dana@example.com
                  refund-triage: run stripe__create_refund
                  amount_usd: 240
                  reason: wrong charge — ch_9001 is the original
```

## Getting started

1. **Define the agent** — `v1.yaml`: its objective, the tools it may call, which of
   them need a human.
2. **Set its policy** — `runtime.yaml` for budgets and authority, `lifecycle.yaml`
   for pause, cooldown and rollback rules.
3. **Package it** — `charter push` seals the version into your registry. Skip it
   while developing: a worker reads a directory just as well.
4. **Configure a worker** — `worker.yaml`: credentials, and which agents and
   versions this process serves.
5. **Apply it** — `charter apply` arms config and policy on the control plane;
   `charter agent create` makes the instance that holds the agent's state.
6. **Run it** — `charter run` starts a task; `charter status` says how it went.
7. **Manage it** — the step that doesn't end: approve what it proposes, watch what
   it spends, add a `v2.yaml` when it should behave differently.

You'll need a BoundFlow control plane you can reach, and an API key for it.

Charter is not on PyPI yet, and it needs BoundFlow's `exp/deepagents-harness`
branch: the governor it runs agents under (`run_governed`, `agent_governor`) and
the shared renderer (`boundflow.cli.output`) are both unreleased. Two checkouts
for now:

```bash
git clone https://github.com/boundflow/boundflow
git -C boundflow checkout exp/deepagents-harness
git clone https://github.com/boundflow/charter

python -m venv .venv
.venv/bin/pip install -e boundflow/sdk/python
.venv/bin/pip install -e charter
```

The branch declares the same version as the published `boundflow` on PyPI while
carrying more API, so `pip install -U boundflow` will silently replace it with a
release that has neither the governor nor the renderer — same version number,
different surface. If imports start failing, check that `pip show boundflow` still
points at your checkout.

### A project

```
.
├── worker.yaml               # which agents this worker serves, pricing, channels
└── leads-finder/
    ├── v1.yaml               # objective, tools, gates — versioned, immutable
    ├── v1/skills/            # procedures for that version, shipped with it
    ├── runtime.yaml          # budgets, limits, authority — policy, not versioned
    └── lifecycle.yaml        # pause, cooldown and rollback rules
```

Only `v1.yaml` is required; a version file plus credentials is enough to run an
agent. The split matters: `v1.yaml` is what the agent *does* and is versioned, so
a rollback restores behavior. Budgets and lifecycle rules are today's guardrails
and stay in force across one.

Skills are the procedures the agent should follow once it gets there — how your
refunds policy works, the runbook a new hire would be handed. Drop them in
`v1/skills/<name>/SKILL.md` and they ship with that version; the layout is
deepagents' own, so skills you already have work unchanged. They live inside `v1/`
because rolling back to v1 should restore the instructions v1 was running with.

### Which agents a worker serves

```yaml
serves:
  - agent: leads-finder       # from ./leads-finder, while you develop
    versions: [1]
  - agent: refund-triage      # from the registry, once it is real
    versions: [1, 2]
    repository: ghcr.io/acme/agents
```

A repository derives one address per version — `<repository>/<agent>:v<N>`, the
address `charter push` writes. List every version a lifecycle rule can roll back
to: a worker that cannot build the old version leaves the control plane
dispatching work nobody can handle.

### Credentials

`worker.yaml` is committed, so nothing secret goes in it. Credentials are `${VAR}`
references resolved from the environment when the worker starts:

```yaml
control_plane:
  endpoint: ${BOUNDFLOW_SERVER_ADDRESS}
  api_key: ${BOUNDFLOW_API_KEY}
  tenant: default

llm:
  provider: anthropic
  api_key: ${ANTHROPIC_API_KEY}

store:
  url: ${CHARTER_STORE_URL}
```

Inference is bring-your-own: your model key stays in the worker environment and
model traffic never reaches the control plane. The CLI reads this same file, so
the control plane is configured once and every command talks to the one your
worker does.

### First run

```bash
charter validate .                    # parse and cross-check every file
charter agent create leads-finder     # bring one instance into existence
charter apply .                       # arm config, policy and pricing
charter worker .                      # in its own terminal — this is the process
```

`create` is separate from `apply` because an instance owns state — its own store,
budget and lifecycle history — so bringing one into existence is a decision a
person makes, not something CI does on their behalf. `apply` is safe to re-run as
often as you like.

`create` prints an instance id; keep it. Commands that act on an agent name an
instance, because the instance is the thing holding the state:

```bash
charter run leads-finder --instance a3f9c012 --topic "..."
```

### Deploying

Locally, `charter worker .` is the whole thing, which is what you want while
iterating — the agent's MCP servers and your logs are right in front of you.

For anything long-lived, run the worker as a container. The reason is isolation
rather than packaging: an agent can declare MCP servers as commands, and the
worker executes them. [deploy/](deploy/) has a Dockerfile that mounts the project
rather than baking it in, so changing an objective is a restart, not a rebuild.

## CLI

Working with configuration:

```bash
charter validate .               # parse and cross-check configuration
charter diff .                   # compare declared and deployed state
charter apply .                  # update config, policy and pricing
charter push <agent> <ref>       # publish a version to an OCI registry
charter worker .                 # run agents in this environment
```

Operating an agent:

```bash
charter agent create <agent>     # bring an instance into existence
charter run <agent> --flags      # start a task
charter agents                   # every agent and what it's doing
charter describe <agent>         # authority, limits, rules, any hold
charter tasks <agent>            # task history
charter status <task-id>         # result, cost and why it stopped
charter audit <agent>            # every governance decision recorded

charter pending <agent>          # the open gate, if it's parked on one
charter approve / reject / answer <id>

charter pause <agent>            # stop it taking work
charter resume <agent> --suspension <id>
charter agent delete <agent>     # destroy an instance and its history
```

Read commands take `--json` for the complete record instead of the curated view.

## Architecture

Charter is an opinionated agent layer built on
[BoundFlow](https://github.com/boundflow/boundflow).

`charter apply` compiles your configuration into workflows and policy on the
BoundFlow control plane. Charter workers run the actual agent loop in your
environment, talking to your MCP servers with credentials that never leave it.

```
                    BoundFlow
                  Control Plane
             state • policy • lifecycle
                       │
                      RPC
                       │
                       ▼
              Your environment
        ┌─────────────────────────┐
        │ Charter worker          │
        │                         │
        │ model ↔ agent loop      │
        │             │           │
        │          MCP tools      │
        └─────────────────────────┘
```

Charter introduces no database or service of its own. If the Charter CLI vanished,
deployed agents would keep running through their workers and the control plane.

See [DESIGN.md](DESIGN.md) for the full configuration reference and the design
decisions behind it, and [examples/](examples/) for complete configurations.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ../boundflow/sdk/python   # your BoundFlow checkout
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

End-to-end tests require a BoundFlow control plane and are excluded by default:

```bash
docker compose -f ../boundflow/docker-compose.dist.yml up -d
export BOUNDFLOW_API_KEY=<...>
pytest tests/e2e
```

They use a real control plane, a real MCP subprocess and real governance gates.
Only the model is faked, so the suite stays deterministic and free.
