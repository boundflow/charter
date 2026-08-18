# Charter

**A fleet of AI agents you can actually let near production — without writing any
code.**

Point an agent at your tools, tell it what you want in plain English, and write down
what it's allowed to do. Charter runs it, stops it at the parts that matter, and
gives you one place to see what all of them are doing.

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
  - name: zendesk
    command: npx
    args: ["-y", "@zendesk/mcp"]
    env: [ZENDESK_API_TOKEN]
    tools:
      - tool: get_ticket
      - tool: close_ticket
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

The agent reads the ticket and the charge, decides a $240 refund is warranted — and
stops, because it cannot issue one. A webhook reaches your finance channel with the
amount and its reasoning. Someone runs:

```bash
charter approve apr_01J8Z --agent refund-triage --reason "third dispute this month"
```

Now the refund happens. The agent sees it succeeded, closes the ticket, and reports
what it did.

> [!WARNING]
> **Pre-alpha.** Everything here is written and unit-tested, but no part of it has
> run against a live control plane or a real model yet. Expect first-contact bugs.

---

## Why you can trust it with real permissions

**A gated tool is never given to the agent.** `approval: always` doesn't mean
"stopped when it tries" — the tool is left out of the agent's toolset entirely. It
can only *name* the tool in a proposal, and a separate step makes the call once a
human approves. Nothing irreversible runs inside the agent's own loop.

**It waits for you, for as long as it takes.** A task that stops for approval is
checkpointed, not held in memory. It can wait thirty minutes, resume on a completely
different machine, and still know what it had found, what it spent, and what you
told it last time. Every agent builder gives you a chat loop; almost none survive
the human going to lunch.

**When it gives up, it tells you why.** Not "it tried a lot" — the specific thing
that went wrong, which is what you'd go fix:

```
a human rejected 3 drafts (max_drafts=3) — the objective or the agent is wrong for this task
stripe.create_refund failed 3 times (max_tool_failures=3) — the integration looks broken
```

**Corrections stick.** Rejecting a proposal with a reason isn't just a "no" — the
reason goes into the agent's next attempt, and into what it remembers on future
tasks. An agent you correct three times has been corrected three times.

## Why it works as a fleet

Thirty agents is a different problem from one agent, and it's the one Charter is
built for.

**Agents react to their own history.** Write the rules once; they run without you:

```yaml
rules:
  - when: { metric: num_failures, threshold: 2 }
    then: { pause: { window: 5 } }              # hold it, I'll look
  - when: { metric: cost, threshold: 5.00 }
    then: { cooldown: { window: 20, seconds: 300 } }
  - when: { metric: approval_rejections, threshold: 3 }
    then: { set_version: { target: 1 } }        # roll back to what worked
```

**Rolling back is real.** Configuration is versioned and version files are
immutable, so `set_version: 1` restores an agent that still exists on disk —
prompt, tools, gates and all. The thing that changed is a file in git with an
author and a diff.

**The limits you wrote are the limits in force.** No rule ever quietly adjusts a
cap behind your back. If `runtime.yaml` says $0.30 a task, that's what's enforced,
and the audit log will show it.

**Every decision is recorded.** Who approved what, when, why, and which rule paused
which agent. Not logs you have to grep — a queryable record, because "prove this
agent did what you say" is a question that eventually gets asked.

## The files

| file | versioned | what it holds |
|---|---|---|
| `agents/<name>/v<N>.yaml` | **yes** | objective, inputs, tools, what counts as done |
| `agents/<name>/runtime.yaml` | no | per-task limits — spend, drafts, questions, tool failures |
| `agents/<name>/lifecycle.yaml` | no | pause / cool down / roll back rules |
| `worker.yaml` | no | which agents a process runs, where approvals go, pricing |

Only the first is required. One `v1.yaml` and a couple of environment variables is
a working agent; the rest are things you add when you want them.

## Running it

```bash
charter validate .                    # parse and cross-check every file
charter apply .                       # create/update agents, policies, pricing
charter worker .                      # the process that runs them

charter run <agent> [--flags]         # start a task
charter tasks <agent>                 # what it's been doing
charter status <task-id>              # result, cost, tools called, approvals
charter approve <id> --agent <name> --reason "..."
charter answer <id> "..." --agent <name>
```

Operating an agent needs only its name and your credentials — no repo checkout. The
person approving a refund at 2am got a webhook, not a git clone.

## Architecture

Charter is a compiler and a worker, not a service. `charter apply` turns your YAML
into workflows and policies on a [BoundFlow](https://github.com/boundflow/boundflow)
control plane; the worker executes them. Charter stores nothing itself — no
database, no new failure domain, and if Charter vanished your agents would keep
running.

Inference is bring-your-own. Your model key lives in your environment, and the
control plane never sees it or your traffic.

See [DESIGN.md](DESIGN.md) for the field reference and the reasoning behind each
decision, and [examples/](examples/) for complete working files.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

The MCP tests spawn a real server process. The fakes alone once passed happily while
a field rename made every tool failure look like a success.

End-to-end tests need a control plane and are excluded by default:

```bash
docker compose -f ../convergeplane/docker-compose.dist.yml up -d
export BOUNDFLOW_API_KEY=<from: ... -mode=provision -name=me>
pytest tests/e2e
```

Real control plane, real MCP subprocess, real gates; only the model is faked, so
they're deterministic and free. Every bug this project has hit came from a
boundary a fake didn't model, which is what these exist to cover.
