# Charter — file formats

Charter is a declarative front end to BoundFlow. A Charter agent is one BoundFlow
workflow running the feedback-loop state machine, with its prompt, tools, and
guardrails supplied by YAML instead of Python.

Four files, with different lifetimes:

| File | What it is | Versioned? | Compiles to |
|---|---|---|---|
| `agents/<name>/v<N>.yaml` | Configuration — objective, inputs, MCP tools | **Yes** | `WorkflowConfig(version=N)` + the agent definition the generic handler runs |
| `agents/<name>/runtime.yaml` | Runtime policy — hard caps per iteration | No | `set_agent_runtime_policy` |
| `agents/<name>/lifecycle.yaml` | Lifecycle policy — how it reacts over time | No | `set_workflow_lifecycle_policy` |
| `worker.yaml` | Which agents + versions this worker process serves | No | `@worker.workflow(type, version=N)` registrations |

Only configuration is versioned, mirroring BoundFlow: `WorkflowConfig.version` is
the versioned thing, and `provision()` re-applies policy unconditionally on every
boot. That split is deliberate — a `SetVersion` rollback should restore the old
*behavior* while leaving today's guardrails in force. Rolling policy back with it
would undo the safety net at exactly the moment it's proving useful.

---

## 1. Configuration — `agents/refund-triage/v1.yaml`

The versioned file. Everything that changes what the agent *does*.

```yaml
apiVersion: charter/v1
kind: AgentConfig

name: refund-triage
version: 1

objective: |
  Resolve the refund request on ticket {{ inputs.ticket_id }}.
  Look up the ticket and the underlying charge before proposing anything.
  If the customer's reason is unclear, ask — do not guess.

model: claude-haiku-4-5

# Validated by `charter invoke` before the request is created; interpolated into
# the objective above. Maps onto invoke_workflow(context=...).
inputs:
  ticket_id: { type: string, required: true }
  max_refund_usd: { type: number, default: 100 }

mcp:
  - name: zendesk
    command: npx @zendesk/mcp          # stdio; runs beside the worker
    env: [ZENDESK_API_TOKEN]           # names only — values come from the worker env
    tools:
      read: [get_ticket, search_tickets]
      act:  [close_ticket]
  - name: stripe
    url: https://mcp.stripe.com        # remote; the worker is still the client
    env: [STRIPE_API_KEY]
    tools:
      read: [get_charge, list_refunds]
      act:  [create_refund]

# What ends a run successfully. The agent emits exactly one of these.
outcome:
  deliverable:
    resolution: { type: string }
  # Proposing an `act` tool parks the run on an approval gate; the tool is executed
  # by a separate operation only on the approved branch. Nothing mutating ever runs
  # inside the agent loop.
  approval:
    prompt: "Refund ${{ propose.args.amount }} on ticket {{ inputs.ticket_id }}?"
    timeout_seconds: 1800
  # The agent can ask a human instead of guessing; the answer folds into history
  # and the loop re-enters.
  ask_human:
    timeout_seconds: 240
```

**Version files are immutable.** A `SetVersion` rollback dispatches operations at
the old version number, and the worker must still be able to build that agent
definition — so `v1.yaml` stays on disk forever and `charter apply` on an edited
config writes `v2.yaml`. Editing a version file in place makes a rollback restore
something that no longer exists.

Tool names are namespaced `server.tool` (`stripe.create_refund`) because names are
the only handle policy has — the control plane sees names and counts, never
arguments or results. Undeclared tools are refused rather than passed to the model,
so an MCP server shipping a new tool can't silently widen the agent's authority.

## 2. Runtime policy — `agents/refund-triage/runtime.yaml`

Hard caps, enforced in the worker, frozen for the whole run at request creation.

```yaml
apiVersion: charter/v1
kind: RuntimePolicy
agent: refund-triage

# Everything a Charter author writes is per *run* — the unit they actually think
# in. One run = one task = one BoundFlow request, however many loop iterations it
# takes internally.
per_run:
  max_iterations: 6
  max_cost_usd: 0.30
  max_llm_calls: 40
  tool_call_limits:
    - tool: stripe.get_charge
      max_calls: 5

# Safety valves against one pathological call, not a budget. Sensible defaults;
# most agents never set these.
limits:
  max_tokens_per_call: 1024
  max_call_seconds: 60
```

**How one number gets enforced twice.** BoundFlow's runtime policy is per agent
invocation and resets every trip around the loop, so Charter sets the BoundFlow
cap *equal to* the per-run budget and separately accumulates the real total across
iterations in run context. BoundFlow's copy becomes a hard in-worker backstop — a
single runaway iteration can't exceed the run budget on its own — while Charter's
accumulator enforces the sum. Both defend the same declared number, so whichever
trips first produces the same ceiling, and a bug in Charter's counter still can't
blow past it by more than nothing.

`max_iterations` is Charter's alone; BoundFlow has no view of the loop.

Exceeding any per-run limit fails the run, which is what feeds the lifecycle rules
below.

## 3. Lifecycle policy — `agents/refund-triage/lifecycle.yaml`

How the agent reacts to its own history across tasks. Not versioned; always live.

**Workflow lifecycle only.** Charter deliberately does not expose BoundFlow's
*agent* lifecycle policy (`SetModel`, `SetMaxLlmCalls`, `SetMaxCostUsd`,
`SetMaxTokensPerCall`). See below for why.

```yaml
apiVersion: charter/v1
kind: LifecyclePolicy
agent: refund-triage

rules:
  # "6 tries per task, and if it fails twice, pause so I can see why it's bad."
  - when: { metric: num_failures, threshold: 2 }
    then: { pause: { window: 5 } }
  - when: { metric: cost, threshold: 5.00 }
    then: { cooldown: { window: 20, seconds: 300 } }
  - when: { metric: approval_rejections, threshold: 3 }
    then: { set_version: 1 }
  - when: { metric: tool_failures, threshold: 3, tool: stripe__create_refund }
    then: { pause: { window: 10 } }
```

Metrics are `num_failures`, `cost`, `num_llm_calls`, `latency`,
`approval_rejections`, `tool_failures`. Actions are `pause`, `cooldown`,
`set_version`. All map to BoundFlow's workflow vocabulary verbatim except
`tool_failures`, which compiles to `TOOL_FAILURE_RATE` — a misnomer in the SDK,
where the engine sums a raw count for one tool rather than a ratio, so
`threshold: 3` means three failed calls. Charter adds no metrics of its own, so
anything expressible in the SDK's workflow policy is expressible in YAML.

`charter schema -o .charter` writes these out as JSON Schema, so an editor
offers the valid metrics rather than you looking them up here.

### Why no agent lifecycle policy

The three actions that adjust caps (`SetMaxLlmCalls`, `SetMaxCostUsd`,
`SetMaxTokensPerCall`) would **break the dual enforcement above**. Charter sets
BoundFlow's runtime cap equal to the declared per-run budget and accumulates the
same number itself; an agent rule mutating BoundFlow's copy server-side leaves the
two enforcers defending different numbers, with only one of them visible in YAML.

`SetModel` breaks something worse: **it changes behavior without a version bump.**
`model` is declared in the versioned config file. If a lifecycle rule can swap it
out-of-band, the agent that's actually running is described by no version on disk,
and `set_version` no longer restores a known state — which is the whole reason
configuration is the versioned artifact.

What's lost is automatic cost adaptation, and it comes back better through the
primitive Charter already has: write a `v2.yaml` with the cheaper model, commit
it, and let a rule move the fleet.

```yaml
  - when: { metric: cost, threshold: 5.00 }
    then: { set_version: 2 }        # v2 = same agent, cheaper model
```

Same outcome as a `SetModel` downgrade, except the thing that changed is a file in
git, the change is auditable, and it uses one mechanism instead of two. Every
lifecycle action Charter exposes acts on the *agent as a managed unit* — hold it,
slow it, or move it to a known version — which is the fleet-management posture the
product is for. Tuning an individual agent's model economics is what the SDK is
for.

## 4. Worker manifest — `worker.yaml`

The worker process is generic — one handler implementing the feedback loop for
every Charter agent. This file tells it which `(workflow_type, version)` pairs to
register and where to find the config for each.

```yaml
apiVersion: charter/v1
kind: Worker

control_plane:
  endpoint: ${BOUNDFLOW_ENDPOINT}
  api_key: ${BOUNDFLOW_API_KEY}

llm:
  provider: anthropic
  api_key: ${ANTHROPIC_API_KEY}

serves:
  - agent: refund-triage
    versions: [1, 2]        # must include every version any run might roll back to
  - agent: ticket-summarizer
    versions: [1]

trace_sink:
  kind: otel
  endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT}
```

At boot the worker loads each listed config version and calls
`worker.workflow(agent, version=N)` plus the four fixed operations
(`execute_act`, `log_rejection`, `log_answer`, and the entry handler) once per
version. `serves` is what makes a worker fleet-manageable: which process can run
which agent is declarative, so you can shard agents across workers, or run a
canary worker holding only `v2` while the fleet stays on `v1`.

`versions` is a list rather than a single number for the rollback reason above —
dropping a version from a worker that a lifecycle rule might roll back to leaves
the control plane dispatching operations nobody can handle.

---

## Compilation

`charter apply -f agents/refund-triage/` is `provision()` with the constants read
from YAML:

1. find-or-create the workflow, type = `name`, `WorkflowConfig(version, invoke_mode)`
2. `set_agent_runtime_policy` from `per_iteration`
3. `set_workflow_lifecycle_policy` from `rules`
4. `activate_workflow`

`set_agent_lifecycle_policy` is never called — Charter leaves it unset, so the
effective runtime policy always equals the base policy declared in YAML.

Idempotent, safe on every boot — same property `provision()` already has.

## Notifications

An approval gate nobody hears about is just a slow timeout, so notification is
load-bearing rather than a nicety.

**One primitive: a signed outbound webhook.** `POST` of a JSON envelope with an
`X-Charter-Signature` HMAC-SHA256 header over the body, keyed by a shared secret.
No Slack/PagerDuty/email integrations — every one of those ingests webhooks, as do
Zapier and n8n, and a URL plus a secret is the only credential material Charter
has to hold. A built-in `slack:` channel is a later convenience, not a v1 need.

**Notification is entirely a worker concern.** Agent configs contain no
notification fields at all — not even a channel name. An agent with an approval
gate *must* notify someone; that isn't an agent-author decision, and making the
config name a channel would point an immutable, committed file at something owned
by whoever deploys the worker.

Routing therefore lives in `worker.yaml`, keyed by agent name — a string, not a
path, so nothing reaches back into an agent repo:

```yaml
# worker.yaml
notifications:
  channels:
    - name: finance
      kind: webhook
      url: ${CHARTER_FINANCE_WEBHOOK}
      secret: ${CHARTER_WEBHOOK_SECRET}
    - name: oncall
      kind: webhook
      url: ${CHARTER_ONCALL_WEBHOOK}
      secret: ${CHARTER_WEBHOOK_SECRET}
  routes:                       # first match wins
    - agent: refund-triage
      channel: finance
    - channel: oncall           # catch-all default
```

`charter apply` warns when an agent declares `act` tools and no route resolves for
it. Payload:

```json
{
  "event": "approval_requested",
  "agent": "refund-triage",
  "version": 1,
  "workflow_id": "wf_...",
  "approval_id": "apr_...",
  "request_id": "req_...",
  "justification": "Refund $240 on ticket 4821?",
  "timeout_seconds": 1800,
  "context": { "ticket_id": "4821" }
}
```

`approval_id` and `workflow_id` are what make it actionable: the recipient decides
with `charter approve <approval_id>` / `charter reject <approval_id>`, which call
`approve_workflow`/`reject_workflow`. Charter v1 ships no approval UI and no
inbound callback endpoint — the webhook is outbound-only and the CLI is how a
decision is made.

Delivery is best-effort with bounded retry. A failed notification never fails the
task: the approval still exists and degrades to its existing timeout branch, which
is a state the state machine already handles. Failures are logged loudly, because
"the gate opened and nobody was told" is exactly the condition an operator needs
to find out about.

### Lifecycle events are not notifiable yet

`paused`, `cooldown`, and `set_version` fire in the scheduler, server-side. The
worker is never told, so Charter has no push path for them — which is unfortunate,
since "your agent just got paused" is the most operationally interesting event the
product produces.

Options, none of them free:

- **Poll the audit log** from the worker or CLI. Works today, no BoundFlow change,
  but it's polling and the latency is the poll interval.
- **Server-side webhooks in BoundFlow** — the right answer, and useful to the SDK
  independently of Charter. Notification becomes a control-plane feature emitting
  on the same events it already writes to the audit log.

Deferred until after the worker path works end to end; the audit log means nothing
is lost in the meantime, only delayed.

## Tool failure

A tool failure is always *recorded* — `tool_failure_counts` feeds the server's
`ToolFailureCounts`, which is what `tool_failure_rate` rules read. `on_failure`
only decides whether the task also stops.

Tools run in two different places, so there are four cases:

| | `on_failure: continue` | `on_failure: fail` |
|---|---|---|
| `approval: never` (inline) | Orchestrator catches, counts, hands the error to the model, loop continues. Pure BoundFlow. | Same, then `entry` checks `result.tool_failure_counts` after `run_agent` → `mark_failed()` + `Complete()` |
| `approval: always` (`execute_act`) | Charter catches, records, folds the error into history, `Next(entry)` so the agent can react | Charter catches, records, `mark_failed()` + `Complete()` |

The bottom row has no orchestrator and no `run_agent` call, so Charter writes the
metrics snapshot by hand into `ctx.agent_state_updates[agent]` — same shape
`AgentGovernor.snapshot()` produces, flushed by the worker into the operation
proto. Without it, an approved tool that fails every time would be invisible to
the very rule meant to catch it.

What counts as a failure at all: BoundFlow counts one when a tool handler
**raises**. MCP reports most tool errors as a *successful* response carrying
`isError: true`, so Charter's wrapper must translate that into an exception — get
this wrong and every MCP failure is recorded as a success, silently disabling
`on_failure`, `tool_failure_rate`, and the whole failure story. A tool returning
ordinary content that merely describes a failure is not detectable as one.

## Runs are tasks

Declared `inputs` become CLI flags, one per input, `snake_case` to `--kebab-case`:

```bash
charter run refund-triage --ticket-id 4821 --max-refund-usd 250
charter run refund-triage --inputs-file ./task.json     # scripted
echo '{"ticket_id":"4821"}' | charter run refund-triage --inputs-file -
```

A missing or unknown flag prints the `inputs` block — types, defaults, enums, and
descriptions — so the config file is the only place a task's interface is written
down. It isn't in `--help`, because Typer builds that before the agent is named. Values are validated and coerced before the request is
created; an unknown flag or a missing required input fails locally, without
burning a run.

All three forms end at `invoke_workflow(context={...})`, returning the request id
as the task id. Anything that can call BoundFlow's SDK or gRPC API directly can
invoke the same agent without Charter in the path — `charter run` adds validation
and a name, not a new mechanism.

A Charter agent already has everything a task primitive needs — typed inputs, a
bounded budget, a declared deliverable, a terminal `Complete(result=...)`. So
"task" is what we *call* a run, not a new construct to build: one invoke = one
task = one BoundFlow request, and `max_iterations` is how many times the agent may
think before the task is declared failed.

That naming is what makes "just trust the agent to go do stuff" coherent rather
than alarming. Unbounded trust has no scope to be bounded by; trust *within a
task* does — the agent is free to reason, retry, and call every `read` tool it
wants, spending up to that task's budget, and the only thing it cannot do
unilaterally is the irreversible edge. Every guardrail in Charter is scoped to a
task, or to a pattern across tasks (lifecycle). There is no third level, which is
why there's no third file.

## What one task actually looks like

From a Charter user's point of view, with no BoundFlow vocabulary.

You wrote two things: an **objective** in plain English, and a **menu of tools**
split into ones the agent may use freely (`read`) and ones it may only ask for
(`act`). You did not write any steps. Then:

```bash
$ charter run refund-triage --ticket-id 4821
task_id: req_01J8Z...
```

1. The agent reads the objective and calls `zendesk.get_ticket`, then
   `stripe.get_charge`. Both are `read` — no gate, no notification, it just works.
2. It decides a $240 refund is warranted. It **cannot call `create_refund`** —
   that tool was never handed to it. All it can do is say so:
   `propose{tool: stripe.create_refund, args: {amount: 240}, why: "..."}`.
3. The task parks. A webhook hits your finance channel with the justification and
   an `approval_id`. Nothing has happened to anyone's money.
4. Someone runs `charter approve apr_...`. Only now does Charter call the tool.
5. The result folds back in and the agent continues. Seeing the refund succeeded,
   it proposes `zendesk.close_ticket` — gated the same way, approved the same way.
6. Finally it submits its **deliverable**: `{resolution: "...", refunded_usd: 240}`.
   The task is complete, and `charter status req_01J8Z...` shows the result, what
   it cost, which tools it called, and who approved what.

### Not every mutation costs a human

The walkthrough above is the strictest setting. Each act tool declares its own
requirement, and so does the deliverable:

```yaml
act:
  - tool: add_internal_note
    approval: never        # inline, no round trip — the model gets the tool
  - tool: close_ticket
    approval: never
  - tool: create_refund
    approval: always       # moves money; a human, every time
```

- **`never`** — the tool goes into `AgentDefinition.tools` and the model calls it
  inline. No proposal, no extra iteration, no notification. Still counted, still
  subject to `tool_call_limits`, still in the trace.
- **`always`** — the model never receives the tool. Proposal-only, parks every time.

Set every act to `never`, `deliverable_approval: never`, and omit `ask_human`, and
you have a fully autonomous agent that never asks anyone anything. It is still
governed: per-task budget, tool limits, the full audit trail, and lifecycle rules
that pause or roll it back when it starts failing. That's the honest version of
"trust the agent" — no human in the loop, but not unsupervised either.

### Confidence is a reasoning tool, not a gate

Approval is deliberately binary, and confidence appears in exactly one place:
`outcome.ask_human.below_confidence`. When set, an agent that rates its own
submission below the bar is bounced back into the loop and told to ask a question
or gather more information rather than proceed.

Authorization is a decision a human makes in advance and writes down. It is never
delegated to the model's opinion of itself — self-reported confidence is poorly
calibrated, and a confidently-wrong model reports high. Wiring it to authorization
would mean miscalibration costs you an unsupervised refund. Wiring it to
*asking* means miscalibration costs you an unnecessary question, and the failure
mode where it doesn't ask just lands back on the normal path: it proposes
something, gated exactly as before.

Same number, same unreliability — pointed somewhere the downside is annoyance
instead of money.

**Where do you specify the final action? You don't — that's the product.** You
specify the menu and the objective; the agent chooses which actions to take and in
what order, and every irreversible one stops for you. If you had to declare the
final action you'd be writing a workflow, and you'd want BoundFlow's SDK instead.

The deliverable is a *report*, not an action — it's how the agent tells you what it
did, and it's what `charter status` shows. If you want an outcome to land somewhere
(post the reply, write the row, page someone), that destination is just another
`act` tool, gated like the rest. There is deliberately no ungated final side
effect, because "the last thing it does" is exactly the thing you'd most want to
approve.

## The state machine the generic handler runs

Unchanged from `question_answerer`, with the receipt-specific parts lifted out:

```
entry -> agent -> deliverable   -> Complete
                \-> propose act -> AwaitApproval -> approved -> execute_act -> entry
                \                                \-> rejected -> AwaitInput("why?") -> entry
                \-> ask_human   -> AwaitInput -> entry
```

**Only a deliverable ends a task.** An approved act is a step, not a terminus: the
tool is called, its result folds into history, and the loop re-enters. That buys
three things — an agent can take several actions in one task (refund, *then* close
the ticket), each separately gated; it observes what its action actually returned
rather than acting blind; and it always gets to report what it did, so the task
result describes the work instead of stopping mid-sentence at the last side effect.

`entry` is the only place that stages context and calls the agent; every re-entry
is "fold in new information, run again." It also owns the `per_run` counters,
since those are the constraint BoundFlow can't enforce for it.

## Field reference

The example files carry no comments — this is where the fields are specified.

### AgentConfig — `agents/<name>/v<N>.yaml`

Versioned and immutable once applied.

| field | type | req | constraints | compiles to |
|---|---|---|---|---|
| `apiVersion` | string | yes | `charter/v1` | — |
| `kind` | string | yes | `AgentConfig` | — |
| `name` | string | yes | `^[a-z][a-z0-9-]{2,62}$`, equals parent dir | `create_workflow(workflow_type=)` |
| `version` | int | yes | `>= 1`, equals `N` in filename | `WorkflowConfig.version` |
| `description` | string | no | human-facing only, never sent to the model | — |
| `model` | string | yes | tenant must have pricing for it | `AgentDefinition.model` |
| `objective` | string | yes | non-empty; `{{ inputs.<name> }}` only | `AgentDefinition.system_prompt` |
| `instructions` | list | no | filenames under `v<N>/`, `.md`, no paths | appended to the system prompt |
| `inputs` | map | no | see below | `invoke_workflow(context=)` |
| `mcp` | list | no | see below | `AgentDefinition.tools` |
| `outcome` | object | yes | see below | `AgentDefinition.output_schema` |

**`instructions`** are markdown documents the agent works from — a refund policy,
an escalation matrix, worked examples. They live in `v<N>/` beside `v<N>.yaml`:

```
agents/refund-triage/
  v1.yaml            instructions: [refund-policy.md]
  v1/refund-policy.md
  v2.yaml
  v2/refund-policy.md   ← a different one; v1's is untouched
```

Versioned by the same rule as the config, which is the point: `set_version: 1`
restores the exact prompt v1 ran with, and editing a version's document in place is
the same mistake as editing its yaml. A missing document fails at `charter
validate`, not a round into the first task.

They're appended to the system prompt, each headed by its filename, and rendered
with the objective — so a document can reference `{{ inputs.* }}` the same way, and
an undeclared reference is caught in the same pass. Being static per version, they
sit in the cached prefix.

What this is *not*: on-demand retrieval. Loading a document at runtime would be a
tool call — in the tool list, against `max_llm_calls`, needing governance — and
Charter already has a name for "the agent fetches knowledge while it works": an MCP
server. `instructions` is what an agent always needs to know; a corpus too large to
inline is a server.

**Templating** is `{{ inputs.<name> }}` and nothing else — no expressions, filters,
conditionals, or loops. An undeclared reference fails at `charter apply`. It is
valid in `objective` and `outcome.approval.note`.

**`inputs.<name>`** — `type` (required: `string|integer|number|boolean`),
`required` (default false), `default` (matching `type`; mutually exclusive with
`required: true`), `enum` (non-empty list; `default` must be a member),
`description`. Objects and arrays are unsupported: inputs are CLI flags. Declaring
any input makes the agent task-shaped, forcing `invoke_mode: queue`.

**`mcp[]`** — `name` (`^[a-z][a-z0-9_]{1,31}$`, unique; namespaces every tool as
`name.tool`), exactly one of `command` (+ optional `args`) or `url` (https only),
`env` (variable *names* only, must exist in the worker environment at boot),
`tools` (non-empty).

**`mcp[].tools[]`** — `tool` (required, unique once namespaced), plus:

| | values | default | meaning |
|---|---|---|---|
| `approval` | `never` \| `always` | unset | `never`: handed to the model, called inline. `always`: the model never receives it; it can only propose it, and a human approves before it runs. Unset follows the server's `approval:` rules, or `never` if it has none. |
| `on_failure` | `continue` \| `fail` | `continue` | `continue`: the model is told and carries on. `fail`: `mark_failed()` + `Complete()`, checked at the next iteration boundary. |

Tools the server exposes but this file does not declare are refused and never
shown to the model.

**`mcp[].approval`** decides gating from what the server says a tool does, instead
of a per-tool entry for each of thirty:

```yaml
approval:
  read_only: never      # read_only_hint: true
  default: always       # everything else, including anything unannotated
```

Resolved at boot from MCP's `ToolAnnotations` — and it can only ever **tighten**. A
server marking something destructive gates it at the next boot with no deploy; a
server marking something read-only changes nothing, because removing a gate is a
decision that belongs in a file with an author. Same rule as `Budget`, which can
narrow what policy allows and never widen it.

Annotations are hints, and the MCP spec says not to make tool-use decisions from
them on an untrusted server. The ratchet is what makes them safe to honour: the
worst a lying server can do is gate something unnecessarily. `charter import`
drafts a config from the same hints, which is where they do the most good — a
review, once, rather than a runtime decision forever.

**`outcome`**:

| field | type | req | notes |
|---|---|---|---|
| `deliverable` | map | yes | non-empty. `<field>: {type, description}`. `propose`, `ask_human`, `confidence` are reserved. Submitting it ends the task. |
| `deliverable_approval` | string | no | `never` (default) \| `always`. Binary, deliberately. |
| `approval` | object | cond | required iff any tool is `approval: always` or `deliverable_approval: always`; forbidden otherwise |
| `ask_human` | object | no | omit to forbid the agent from asking |

`approval` — `timeout_seconds` (default 1800), `on_timeout` (`reject` default \|
`fail`), `note` (optional extra context, `{{ inputs.* }}` only).

Charter composes `AwaitApproval.justification` from three sources, so no single
author or model omission can leave an approver flying blind:

| shown to the approver | comes from |
|---|---|
| the tool and its arguments | `propose.tool`, `propose.args` — the agent's proposal |
| why the agent wants it | `propose.why` — the agent's own reasoning |
| what task this is | `outcome.approval.note` — the author, `{{ inputs.* }}` |

Only the last is templatable. The first two are Charter's rendering, so a gate
cannot be authored with the amount left out. For `deliverable_approval: always`
it's the same field with the deliverable as subject, plus the note.

`ask_human` — `timeout_seconds` (default 240), `on_timeout` (`continue` default \|
`fail`), `below_confidence` (optional, 0..1). Setting `below_confidence` injects a
required `confidence` field into the output schema; a submission rating itself
below the bar is bounced back into the loop and told to ask or gather more
information. This is the only place confidence appears in Charter.

There is deliberately no `prompt` here: the question is `AwaitInput(prompt=...)`,
supplied by the agent at runtime from the generated `ask_human.question` field. You
can't author it in advance because you don't know what the agent will get stuck on.

Charter generates two output-schema branches the file never declares: `propose`
(`{tool, args, why}`, injected iff any tool is `approval: always`) and `ask_human`
(`{question}`, injected iff `outcome.ask_human` is set).

### RuntimePolicy — `agents/<name>/runtime.yaml`

Not versioned. Purely quantitative — nothing here changes the agent's shape.

| field | type | req | notes |
|---|---|---|---|
| `agent` | string | yes | matches the config's `name` |
| `per_run.max_iterations` | int | — | `> 0`. Charter-only; BoundFlow can't see the loop. |
| `per_run.max_cost_usd` | float | — | `> 0` |
| `per_run.max_llm_calls` | int | — | `> 0` |
| `per_run.tool_call_limits[]` | list | no | `{tool, max_calls}`; `tool` must be declared in the config |
| `limits.max_tokens_per_call` | int | no | default 1024 |
| `limits.max_call_seconds` | float | no | default 60 |

At least one of the three `per_run` budgets is required. Each is enforced twice
against the same number: Charter accumulates the real total across iterations, and
sets BoundFlow's per-invocation `RuntimePolicy` to the same value as an in-worker
backstop. `limits` are valves against one pathological call, not budgets.

### LifecyclePolicy — `agents/<name>/lifecycle.yaml`

Not versioned. Workflow-level only — Charter never sets an agent lifecycle policy,
so the effective runtime policy always equals what `runtime.yaml` says.

| `when.metric` | `when.tool` | notes |
|---|---|---|
| `num_failures` | — | includes budget-exceeded and `on_failure: fail` |
| `cost` | — | USD |
| `num_llm_calls` | — | |
| `latency` | — | seconds |
| `approval_rejections` | — | |
| `tool_failures` | required | a summed count of failed calls to that tool |

Charter's `tool_failures` compiles to BoundFlow's `WorkflowMetric.TOOL_FAILURE_RATE`,
which is a misnomer in the SDK: the lifecycle engine reads `ToolFailureCounts[tool]`
and compares a summed *count*, never a ratio. Charter uses the accurate name and
translates at compile time, so `threshold: 3` means three failed calls. Every other
metric name is BoundFlow's verbatim.

`when.threshold` is summed over the action's window and compared `>=`. `then` is
exactly one of `pause: {window}`, `cooldown: {window, seconds}`, or
`set_version: {target}`. A `set_version` target must exist on disk *and* appear in
`serves[].versions` for every worker running the agent.

### Worker — `worker.yaml`

Not versioned. Every secret is an `${ENV_VAR}` reference, never a literal.

| field | req | notes |
|---|---|---|
| `name` | no | defaults to hostname |
| `control_plane` | yes | `endpoint`, `api_key`, `tenant_id` |
| `llm` | yes | `provider` (`anthropic` \| `langchain`), `api_key` |
| `agents_dir` | no | default `./agents`, relative to this file |
| `serves[]` | yes | `{agent, versions: [int]}` — every version any rollback might target |
| `notifications.channels[]` | no | `{name, kind: webhook, url, secret?, timeout_seconds=5, max_attempts=3}` |
| `notifications.routes[]` | no | `{agent?, events?, channel}`, first match wins; omit `agent` for a catch-all |
| `trace_sink` | no | `{kind: none\|logging\|jsonl\|otel, path?, endpoint?}` |
| `model_pricing` | no | `<model>: {input, output}` per 1M tokens; tenant-global |

`serves[].agent` is the only reference between the two artifacts, and it points one
way, by name. Agent configs never reference a worker.

## The CLI

One CLI. `charter` is a sibling of the `boundflow` CLI, not a layer on top: both
are Typer apps calling `ControlPlaneClient` over gRPC. Charter never shells out to
`boundflow` — that would mean Python → subprocess → Python → gRPC, turning validated
pydantic models into `--tool-limit TOOL:MAX` strings and typed exceptions into exit
codes.

Charter must cover the whole surface a Charter user needs, because reaching for the
`boundflow` CLI is not neutral. Reads there technically work — a Charter agent is an
ordinary workflow — but they show workflow UUIDs and operation names like
`execute_act`, which mean nothing to someone who wrote YAML. And several writes
break Charter's invariants outright:

| command | what it breaks |
|---|---|
| `policy set-agent-lifecycle` | "the effective policy always equals `runtime.yaml`" — the promise the lifecycle design rests on |
| `policy set-agent-runtime` | live policy drifts from YAML; the next `charter apply` silently reverts it |
| `workflow set-version` | can move an agent to a version no worker serves — the outage the loader exists to prevent |

"Mostly works, and some of it quietly corrupts your guarantees" is worse than
"doesn't work." So the surface is:

| command | does |
|---|---|
| `charter validate [path]` | parse and cross-check every file; no network |
| `charter apply [path]` | validate, then create/update workflow, policies, pricing; idempotent |
| `charter run <agent> [--flags]` | validate inputs, `invoke_workflow`, print the task id |
| `charter tasks <agent>` | recent tasks with outcome, cost, duration |
| `charter status <task-id>` | result, cost, tools called, approvals, why it stopped |
| `charter pending [agent]` | open approval and input gates |
| `charter approve <id> [--reason]` | resolve to workflow + approval id, decide |
| `charter reject <id> [--reason]` | same |
| `charter answer <id> <text>` | respond to an `ask_human` gate |
| `charter pause/resume <agent>` | hold or release the fleet member |
| `charter rollback <agent> --to N` | manual `set_version`, refusing targets no worker serves |
| `charter memory <agent>` | print exactly what the agent is shown from the audit log |
| `charter worker [-f worker.yaml]` | run the generic worker process |

The `boundflow` CLI stays available as a debugging escape hatch, and it's worth
keeping that true — if `boundflow workflow runs <id>` can't inspect a Charter agent,
we've built a walled garden. But it isn't part of the product surface, and the
policy-writing commands should be documented as unsafe against Charter-managed
workflows.

## Asks of BoundFlow

### Pagination on ListWorkflowRuns

`list_workflow_runs` returns **every** run a workflow has ever had — no `LIMIT` in
the query, no page token, and nothing trims them. `charter tasks` therefore fetches
an entire history to show twenty rows, and filters locally.

Fine at ten runs, wrong at ten thousand, and a periodic agent gets there in a
month. The ask is keyset pagination — `limit` plus an `after` cursor on
`(created_at, request_id)` — rather than offset, which skips and repeats rows when
new runs land mid-scan, exactly when you'd be paging. A server-side outcome filter
would help too: "show me the failures" is the question people actually have.

Related: `WorkflowMetrics` is scoped to the workflow's **current version** while
the run list spans every version, so after a `set_version` rollback the totals a
lifecycle rule judges drop to that version's record while the history still shows
everything. Defensible, but neither is labelled as such.


Three related changes, all in the approval path. Together they make the audit log
self-describing and remove a branch from Charter's state machine.

1. **Persist `justification` into `ApprovalAuditDetails`.** It's written to the job
   row at `ParkForApproval` and cleared when the decision lands, so the permanent
   record says "rejected" with no subject.
2. **Accept an optional `reason` on `approve_workflow` / `reject_workflow`**, stored
   alongside the decision. Approvals benefit too: "fine, but stop refunding
   shipping" is the highest-quality signal the system produces and has nowhere to go.
3. **Surface it on the resumed operation** — `ctx.approval_reason`, beside the
   existing `ctx.input_answer`. Without this the reason is recorded but Charter must
   fetch it back over gRPC to use it.

With all three, the rejection path collapses from

```
rejected -> ask_rejection_reason -> AwaitInput("why?") -> log_rejection_reason -> entry
```

to `rejected -> entry`, deleting two operations, a second park, a second human
interaction, and a timeout branch. It also removes a failure mode: today a human
can reject and walk away, the "why?" input times out, and the agent redrafts having
learned nothing from a rejection that did happen.

## Open

- **Scheduling.** `WorkflowConfig.repeat_every_seconds` and `triggerable` aren't
  in any file above. Probably a `schedule:` block in configuration — but a
  periodic agent can't take per-invocation `inputs`, so the two are mutually
  exclusive and the schema should say so.
- **`invoke_mode`.** Derived, not authored: an agent that declares `inputs:` is
  doing discrete tasks and must be `queue` (coalescing would silently discard a
  ticket). An agent with no inputs is being told "something changed, go look", and
  `coalesce` is correct — two such triggers really are one piece of work.
- **Model pricing.** `set_model_pricing` is per-tenant and global, not per-agent —
  belongs in `worker.yaml` or a separate tenant-level file, not in an agent's config.
