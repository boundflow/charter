# Charter

An autonomous agent shouldn't just have a prompt and tools. It should have a
charter — a defined responsibility, explicit authority, an operational budget, and
rules for when humans or the platform take control.

Charter turns that contract into a persistent agent you can deploy and leave
running in your own environment.

Give it an objective and the MCP tools it needs. Define what it may do
autonomously, what requires human approval, and how much it may spend. Charter
handles durable human escalation while it works, tracks its behavior across tasks
and versions, and can pause, cool down, or roll back an agent when that behavior
crosses the thresholds you set.

The agent runs in your environment. The control plane governs it across time.

## Define an agent

A Charter agent is declarative. Its versioned specification defines what the agent
is responsible for and the capabilities it receives:

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

Deploy it and give it work:

```bash
charter apply agents/refund-triage/
charter run refund-triage --ticket-id 4821
```

Suppose the agent investigates the ticket and charge and concludes that a $240
refund is warranted.

It cannot issue the refund.

`create_refund` requires approval, so Charter parks the task and surfaces the
proposed action and the agent's reasoning to a human.

```bash
charter approve apr_01J8Z \
  --agent refund-triage \
  --reason "third dispute this month"
```

Only then is the refund executed. The agent receives the result, finishes the task,
and reports what happened.

Close your laptop in between. The task doesn't live there.

> [!WARNING]
> **Pre-alpha.** A first Charter agent has run end-to-end against a live BoundFlow
> control plane, a real model, and a real MCP server. Plenty hasn't. Expect rough
> edges and expect the configuration format to change.

## Explicit authority

Giving an agent access to an MCP server does not give it access to everything that
server exposes.

Charter only shows the model tools you explicitly declare:

```
mcp stripe: 34 tools available, 2 declared (32 ignored)
```

If Stripe exposes 34 tools and your Charter declares two, the agent sees two. The
remaining 32 are outside its authority. If the server adds another tool tomorrow,
your agent doesn't silently gain another capability.

Approval-gated tools have an even stronger boundary.

```yaml
- tool: create_refund
  approval: always
```

`create_refund` is not available for autonomous execution inside the agent loop.
The agent may propose using it, but a separate execution step performs the action
only after human approval.

The distinction is intentional:

**The model can propose authority it doesn't have. It cannot grant that authority
to itself.**

## An operational budget

Autonomy needs limits beyond permissions.

Charter can bound the resources an agent may consume while completing a task:

```yaml
per_run:
  max_cost_usd: 0.30
  max_llm_calls: 40
  max_drafts: 3
  max_questions: 2
  max_tool_failures: 3
```

Those limits hold across the entire task — across reasoning rounds, retries, human
feedback, and tool calls.

When a limit is exhausted, Charter reports the operational reason rather than
collapsing everything into a generic agent failure:

```
a human rejected 3 drafts (max_drafts=3)
the objective or the agent may be wrong for this task

stripe__create_refund failed 3 times (max_tool_failures=3)
the integration looks broken
```

`max_tool_failures` is per tool, so one broken integration can trip its circuit
breaker without consuming an aggregate failure budget for unrelated tools.

## Humans are part of the runtime

There are two reasons an autonomous agent may need to stop and involve someone.

It can ask a question when it lacks information it shouldn't guess.

It can request approval when the action it wants to take exceeds its delegated
authority.

Both are durable interruptions:

```bash
charter pending refund-triage

charter approve apr_01J8Z \
  --agent refund-triage \
  --reason "confirmed duplicate"

charter answer inp_01J8Z \
  "use the March charge" \
  --agent refund-triage
```

A waiting task isn't a Python process sitting around hoping someone comes back.

Charter checkpoints it. The task can wait thirty minutes or overnight and resume on
another worker while retaining what it discovered, what it spent, and what the
human told it.

Rejection is also feedback, not just a failed gate. Reject an action or proposed
result with a reason and the agent receives that reason when it resumes, giving it
a chance to revise its approach.

## The platform has authority too

Human approval governs individual actions.

Lifecycle policy governs the agent itself.

Charter tracks operational evidence across tasks and can act when an agent's
behavior stops meeting the contract you've defined:

```yaml
rules:
  - when: { metric: num_failures, threshold: 2 }
    then: { pause: { window: 5 } }

  - when: { metric: cost, threshold: 5.00 }
    then: { cooldown: { window: 20, seconds: 300 } }

  - when: { metric: approval_rejections, threshold: 3 }
    then: { set_version: { target: 1 } }
```

That creates two distinct control boundaries:

```
Agent wants to exceed its authority
              ↓
          HUMAN ACTS

Agent's behavior degrades across tasks
              ↓
         PLATFORM ACTS
```

A noisy agent can be cooled down. A repeatedly failing agent can be paused. A new
version whose decisions are repeatedly rejected can be rolled back to the version
that was working.

And rollback means more than swapping a prompt string. Agent versions are immutable
specifications: objective, tools, gates, and other behavior-producing configuration
move together.

The thing that changed is a file in Git, with an author and a diff.

Charter does not silently rewrite the limits you've declared. Runtime and lifecycle
policy remain explicit configuration, and `charter diff` shows whether what's
running matches what's in your files.

## Operate agents, not sessions

A Charter agent persists beyond any individual task.

```
$ charter agents

AGENT           VER  STATUS  ACTIVITY
invoice-chaser  v1   paused  idle
refund-triage   v1   active  awaiting_approval
ticket-sweeper  v1   active  idle
```

Inspect the operational contract and how close lifecycle rules are to firing:

```
$ charter describe refund-triage

limits per task
  max_cost_usd    0.25
  max_llm_calls   20

rules
  num_failures   1 of 2     -> pause window=5
  cost           5.2 of 5   -> cooldown window=20 seconds=300
```

And afterwards, reconstruct what happened:

```
$ charter audit refund-triage

2026-08-18 03:47  approval rejected by dana@example.com
                  refund-triage: run stripe__create_refund
                  amount_usd: 240
                  reason: wrong charge — ch_9001 is the original
```

The agent isn't just a loop that ran. It's an operational resource with a current
version, state, history, authority, and behavior over time.

## Getting started

Assumes a BoundFlow control plane you can reach and an API key for it.

```bash
pip install charter
```

### Credentials

`worker.yaml` is committed, so nothing secret is written in it. Every credential
is a `${VAR}` reference resolved from the environment when the worker starts:

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

```bash
export BOUNDFLOW_SERVER_ADDRESS=http://localhost:50051
export BOUNDFLOW_API_KEY=...      # the control plane
export ANTHROPIC_API_KEY=...      # inference — never reaches the control plane
export CHARTER_STORE_URL=postgresql://...   # the agent's checkpoints and files
```

The CLI reads the same block, from `worker.yaml` in the working directory or
wherever `CHARTER_PROJECT` points. So the control plane is configured once, and
`charter describe` talks to the same one `charter worker` does. With no manifest
to hand — on call, with credentials and an agent name but no checkout — it falls
back to `BOUNDFLOW_*` from the environment.

### A project

```
.
├── worker.yaml               # which agents this worker serves, pricing, channels
└── leads-finder/
    ├── v1.yaml               # objective, tools, gates — versioned, immutable
    ├── runtime.yaml          # budgets, limits, authority — policy, not versioned
    └── lifecycle.yaml        # pause, cooldown and rollback rules
```

Only `v1.yaml` is required.

### First run

```bash
charter validate .                    # parse and cross-check every file
charter agent create leads-finder     # bring one instance into existence
charter apply .                       # arm config, policy and pricing
```

`create` and `apply` are separate on purpose. An instance owns state — its own
store, budget and lifecycle history — so bringing one into existence is a decision
someone makes, not something re-running config in CI does on their behalf. `apply`
is then safe to run as often as you like, and is what arms model pricing from
`worker.yaml`. **Skip it and runs cost $0.00**, so cost limits never trigger.

`create` prints the instance id. Keep it — every command that acts on an agent
names one:

```bash
charter worker .                              # its own terminal; this is the process
charter run leads-finder --instance a3f9c012 --topic "..."
```

An agent name addresses a *kind* of instance. The instance is the entity that
holds the state — its conversation, its budget, its lifecycle history — so a name
alone is never a target, including when there is only one. Charter resolved the
name in that case for a while, which was pleasant right up until someone created a
second instance and every command they had learned meant something else. Forget
the id and the error lists what exists with the flag filled in.

`--all` fans out where that is meaningful — `charter apply --all` is how CI
reconciles config without knowing ids.

### While it runs

```bash
charter agents                   # every agent and what it is doing
charter describe <agent>         # authority, armed limits, rules, any hold
charter tasks <agent> [--failed] # task history
charter status <task-id>         # result, cost, and why it stopped
charter pending <agent>          # the open gate, if it is parked on one
charter approve <id> / reject <id> / answer <id> "..."
```

All of these take `--instance <id>`, and `--json` for the complete record rather
than the curated view.

### Changing things

Edit `runtime.yaml` and `charter apply` — limits are read per operation, so a
tightened budget lands on the next round without restarting a worker. Behaviour is
versioned instead: add a `v2.yaml` and apply, and existing tasks finish on the
version they started. `charter diff .` compares what is armed against your files.

### Deploying

Locally, `charter worker .` is the whole thing — which is what you want while
iterating, since the agent's MCP servers and your own logs are in front of you.

For anything long-lived, run the worker as a container. The reason is isolation
rather than packaging: an agent declares MCP servers as commands, and the worker
executes them. [deploy/](deploy/) has a Dockerfile that mounts the project rather
than baking it in, so changing an objective is a restart and not a rebuild.

## Configuration

Charter separates the agent itself from the operational policy around it:

| File | Versioned | Contains |
|---|---|---|
| `agents/<name>/v<N>.yaml` | yes | objective, inputs, tools, gates, outcome |
| `agents/<name>/runtime.yaml` | no | per-task budgets and limits |
| `agents/<name>/lifecycle.yaml` | no | pause, cooldown, and rollback rules |
| `worker.yaml` | no | workers, approval channels, pricing |

Only the agent specification is required.

A `v1.yaml` and the required environment variables are enough to run an agent.
Runtime budgets, lifecycle rules, and additional operational controls can be added
as needed.

## CLI

Configuration, and the agent artifact — Charter's own:

```bash
charter validate .               # parse and cross-check configuration
charter diff .                   # compare declared and deployed state
charter apply .                  # update config, policy and pricing
charter push <agent> <ref>       # publish a version to an OCI registry
charter worker .                 # run agents in this environment
charter schema                   # JSON Schema for the config files
```

Operating an agent. These address agents by name and delegate to the control
plane, so anything here can also be done with `boundflow` against a workflow id:

```bash
charter agent create <agent>     # bring an instance into existence
charter run <agent> --instance <id> [--flags]   # start a task
charter agents                   # list agents and current activity
charter describe <agent>         # authority, limits, rules, any hold
charter tasks <agent> [--failed] # task history
charter status <task-id>         # result, cost and why it stopped
charter audit <agent>            # every governance decision recorded

charter pending <agent>                  # the open gate, if parked on one
charter approve <id> [--reason ...]      # approve a parked gate
charter reject <id> [--reason ...]       # reject one
charter answer <id> "..."                # answer a question

charter pause <agent> [--now]            # stop it taking work; prints a hold id
charter resume <agent> --suspension <id> # release that hold
charter abandon <agent> [--all]          # drop queued tasks, irreversibly
charter agent delete <agent>             # destroy an instance and its history
```

`--json` on any read command prints the complete record instead of the curated
view. `charter pause` prints the id identifying the hold as yours, and `resume`
requires it — releasing whichever hold happens to be there is how one operator
silently undoes another's.

## Architecture

Charter is an opinionated agent layer built on
[BoundFlow](https://github.com/boundflow/boundflow).

`charter apply` compiles the declarative agent configuration into workflows and
policy on the BoundFlow control plane. Charter workers execute the actual agent
loop in your environment and connect to your MCP servers using credentials that
stay there.

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

Charter itself introduces no separate database or service. If the Charter CLI
disappeared, deployed agents would continue running through their workers and the
BoundFlow control plane.

Inference is bring-your-own. Model credentials stay in the worker environment; the
control plane does not need them or the model traffic.

See [DESIGN.md](DESIGN.md) for the configuration reference and design decisions,
and [examples/](examples/) for complete configurations.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ../convergeplane/sdk/python
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

End-to-end tests require a BoundFlow control plane and are excluded by default:

```bash
docker compose -f ../convergeplane/docker-compose.dist.yml up -d
export BOUNDFLOW_API_KEY=<...>
pytest tests/e2e
```

The end-to-end suite uses a real control plane, real MCP subprocess, and real
governance gates. Only the model is faked so the tests remain deterministic and
free.
