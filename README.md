# Charter

**Declarative, governed agents on [BoundFlow](https://github.com/boundflow/boundflow).**

> [!WARNING]
> **Pre-alpha.** Everything here is written and unit-tested, but no part of it has
> yet run against a live control plane or a real model. Expect first-contact bugs.

You give an agent an objective in plain English, a menu of MCP tools, and a policy.
Charter compiles that into a BoundFlow workflow and runs it under governance. No
Python — the agent is YAML.

```yaml
apiVersion: charter/v1
kind: AgentConfig

name: refund-triage
version: 1
model: claude-haiku-4-5

objective: |
  Resolve the refund request on ticket {{ inputs.ticket_id }}.
  Never propose a refund above ${{ inputs.max_refund_usd }}.

inputs:
  ticket_id: { type: string, required: true }

mcp:
  - name: stripe
    url: https://mcp.stripe.com
    env: [STRIPE_API_KEY]
    tools:
      - tool: get_charge
      - tool: create_refund
        approval: always        # the model never receives this tool

outcome:
  deliverable:
    resolution: { type: string }
```

```bash
charter apply agents/refund-triage/
charter run refund-triage --ticket-id 4821
charter approve apr_01J8Z --agent refund-triage --reason "third dispute this month"
```

## What makes it a Charter agent

**The model cannot call a gated tool.** `approval: always` means the tool is never
placed in the agent's toolset — it can only *name* it in a proposal, and a separate
operation makes the call after a human approves. That's a structural property, not
a runtime check.

**Only a deliverable ends a task.** An approved action is a step: the tool runs, its
result folds into history, and the loop re-enters. So an agent can take several
actions in one task, each separately gated, and always gets to report what it did.

**Limits name a cause.** Running out doesn't say "it went round a lot" — it says
which thing went wrong:

```
a human rejected 3 drafts (max_drafts=3) — the objective or the agent is wrong for this task
stripe.create_refund failed 3 times (max_tool_failures=3) — the integration looks broken
```

Those failures feed `num_failures`, so a lifecycle rule pauses the agent on its own.

**The effective policy always equals your YAML.** Charter never sets a BoundFlow
*agent lifecycle* policy, so no rule ever silently changes a declared cap. Fleet
actions — pause, cooldown, roll back to a version — act on the agent as a unit.

**Rollback is a file in git.** Configuration is the versioned artifact; version
files are immutable, so `set_version: 1` restores something that still exists.

## The files

| file | versioned | is |
|---|---|---|
| `agents/<name>/v<N>.yaml` | **yes** | objective, inputs, tools, outcome |
| `agents/<name>/runtime.yaml` | no | per-task limits |
| `agents/<name>/lifecycle.yaml` | no | pause / cooldown / rollback rules |
| `worker.yaml` | no | which agents a worker serves, routing, pricing |

Only the first is required — one `v1.yaml` plus `BOUNDFLOW_*` in the environment is
a working agent. See [DESIGN.md](DESIGN.md) for the field reference and the
reasoning behind each decision, and [examples/](examples/) for complete files.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

The MCP tests spawn a real server process — the fakes alone once passed while a
field rename made every tool failure look like a success.
