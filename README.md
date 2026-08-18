# Charter

**An autonomous agent shouldn't just have a prompt and tools. It should have a
charter** — a defined responsibility, a bounded set of powers, an operational
budget, and rules for when humans or the platform take control.

Charter turns that contract into a persistent agent you can deploy and leave
running in your own environment. It handles escalation and approvals while the
agent works, and tracks its behavior across many tasks and versions. When its own
metrics cross the thresholds you set, Charter pauses it, cools it off, or rolls it
back.

Your agents are config, not code.

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
        approval: always        # the agent is never handed this tool

outcome:
  deliverable:
    resolution: { type: string }
    refunded_usd: { type: number }
```

```bash
charter apply agents/refund-triage/
charter run refund-triage --ticket-id 4821
```

> [!WARNING]
> **Pre-alpha.** A first agent has run end to end against a live control plane, a
> real model and a real MCP server. Plenty has not: expect rough edges, and expect
> the file formats to move.

---

## Explicit authority

An agent can only act with the authority you explicitly give it.

Charter exposes only the tools you declare. If an MCP server provides 34 tools and
you authorize 2, the model sees 2 — the other 32 don't exist from the agent's
perspective. New tools added by the server don't silently expand that authority.

For actions requiring approval, the agent isn't given the tool at all. It can
propose the action, but execution happens separately only after a human approves
it. That means consequential actions never run from inside the autonomous agent
loop.

## An operational budget

Declared per task, and it holds across every round the agent takes and every retry
underneath:

```yaml
per_run:
  max_cost_usd: 0.30
  max_llm_calls: 40
  max_drafts: 3             # submissions a human may reject before giving up
  max_questions: 2          # times it may come back to you before showing you something
  max_tool_failures: 3      # per tool — a circuit breaker, not an aggregate
```

When one runs out, the failure names the cause, because that's what you'd go fix:

```
a human rejected 3 drafts (max_drafts=3) — the objective or the agent is wrong for this task
stripe__create_refund failed 3 times (max_tool_failures=3) — the integration looks broken
```

## Escalation that survives the wait

Two different interruptions. The agent **asks for approval** when it wants to do
something it isn't allowed to do, and **asks a question** when it doesn't know
something and won't guess. You answer either from anywhere:

```bash
charter pending refund-triage
charter approve apr_01J8Z --agent refund-triage --reason "confirmed duplicate"
charter answer inp_01J8Z "use the March charge" --agent refund-triage
```

A parked task is checkpointed, not held in memory. It can wait thirty minutes,
resume on a different machine, and still know what it found, what it spent, and
what you told it. Every agent framework gives you a loop; few survive the human
going to lunch.

Rejecting with a reason isn't just a "no" — the reason goes into the agent's next
attempt, and into what it remembers on later tasks.

## Rules for when the platform takes control

Written once, applied without you:

```yaml
rules:
  - when: { metric: num_failures, threshold: 2 }
    then: { pause: { window: 5 } }                   # hold it, I'll look
  - when: { metric: cost, threshold: 5.00 }
    then: { cooldown: { window: 20, seconds: 300 } } # back off, then resume
  - when: { metric: approval_rejections, threshold: 3 }
    then: { set_version: { target: 1 } }             # go back to what worked
```

Rolling back is real because configuration is versioned and version files are
immutable — `set_version: 1` restores an agent that still exists on disk, prompt
and tools and gates included. The thing that changed is a file in git with an
author and a diff.

And the limits you wrote are the limits in force: Charter never sets a BoundFlow
agent-lifecycle policy, so no rule quietly adjusts a cap behind your back.
`charter diff` proves it against your files.

## You can see all of it

What each agent is doing, and which ones need you:

```
$ charter agents
AGENT           VER  STATUS  ACTIVITY
invoice-chaser  v1   paused  idle
refund-triage   v1   active  awaiting_approval
ticket-sweeper  v1   active  idle
```

What one is allowed to do, and how close its rules are to firing — from credentials
and a name, no checkout:

```
$ charter describe refund-triage
limits per task
  max_cost_usd    0.25
  max_llm_calls   20

rules
  num_failures   1 of 2     -> pause window=5
  cost           5.2 of 5   -> cooldown window=20 seconds=300
```

And afterwards, who decided what, in their words, and which rule fired:

```
$ charter audit refund-triage
2026-08-18 03:47  approval rejected by dana@example.com
                  refund-triage: run stripe__create_refund  amount_usd: 240
                  reason: wrong charge — ch_9001 is the original
```

## The files

| file | versioned | what it holds |
|---|---|---|
| `agents/<name>/v<N>.yaml` | **yes** | objective, inputs, tools, what counts as done |
| `agents/<name>/runtime.yaml` | no | per-task budget and limits |
| `agents/<name>/lifecycle.yaml` | no | pause / cool down / roll back rules |
| `worker.yaml` | no | which agents a process runs, where approvals go, pricing |

Only the first is required. One `v1.yaml` and a couple of environment variables is a
working agent; the rest are things you add when you want them.

## Running it

```bash
charter validate .               # parse and cross-check every file
charter diff .                   # is what's running what you declared?
charter apply .                  # create or update agents and policy
charter worker .                 # the process that runs them

charter run <agent> [--flags]    # start a task
charter agents                   # what every agent is doing
charter describe <agent>         # limits, rules, what's waiting
charter tasks <agent> [--failed] # history
charter status <task-id>         # result, cost, actions taken
charter pending <agent>          # the open gate
charter resume <agent>           # release one a rule paused
```

## Architecture

Charter is a compiler and a worker, not a service. `charter apply` turns your YAML
into workflows and policy on a [BoundFlow](https://github.com/boundflow/boundflow)
control plane; the worker runs them in your environment. Charter stores nothing
itself — no database, no new failure domain, and if Charter vanished your agents
would keep running.

Inference is bring-your-own. Your model key lives in your environment, and the
control plane never sees it or your traffic.

See [DESIGN.md](DESIGN.md) for the field reference and the reasoning behind each
decision, and [examples/](examples/) for complete working files.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ../convergeplane/sdk/python   # unreleased SDK, for now
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

End-to-end tests need a control plane and are excluded by default:

```bash
docker compose -f ../convergeplane/docker-compose.dist.yml up -d
export BOUNDFLOW_API_KEY=<from: ... -mode=provision -name=me>
pytest tests/e2e
```

Real control plane, real MCP subprocess, real gates — only the model is faked, so
they stay deterministic and free. Every bug this project has hit came from a
boundary a fake didn't model, which is what those exist to cover.
