# Charter

**Build and operate production-safe agents that run in your own environment.**

![The Charter console: the fleet with an agent parked on an approval, the decision waiting on a human, and the policy and run history behind it](docs/console.gif)

Charter is an open-source platform for running AI agents in production. You define
an agent and its policies in YAML. Charter runs it on your compute and governs it
from a persistent control plane.

## Why Charter

- **Durable execution.** A run parks for a human and resumes days later on another
  worker, with its state intact.
- **Policy that acts.** Set thresholds on the metrics an agent produces. One that
  crosses a threshold pauses, cools down, or rolls back to the version that worked.
- **Declared authority.** The model sees only the tools you list, gated tools
  require human approval, and budgets cap what a task may spend.
- **Fleet operations.** Every agent's operational state, run history, metrics and
  open decisions, from the CLI or the console.
- **Audit and traces.** Every approval, rejection and policy action is recorded
  with who took it and why. Model and tool calls export as OpenTelemetry GenAI
  traces, which Jaeger, Tempo, Datadog and Langfuse can read.
- **Your network, your data.** Workers run in your environment, so agents reach
  internal services and databases directly. Model keys and prompts never reach the
  control plane, and the agent's conversation, files and traces stay in stores you
  run. Self-host the control plane and use a local model to run Charter fully
  air-gapped.

[DESIGN.md](DESIGN.md) documents every field.

## Status

Charter is pre-1.0, so the configuration format and CLI can still change between
releases. [Feedback](https://github.com/boundflow/charter/issues) is welcome.

## Quickstart

For native model, tool, and subagent history in Langfuse, see
[Langfuse tracing](docs/langfuse.md).

Run your first agent in 5 minutes.

```bash
pip install boundflow-charter          # add [ui] for the console, [otel] for traces
pip install --pre boundflow-charter    # or whatever main is, published every green build
```

### A control plane

Charter needs one to run agents against. To run one locally:

```bash
curl -sSLO https://raw.githubusercontent.com/boundflow/charter/main/deploy/local.compose.yml
docker compose -f local.compose.yml up -d --wait
docker compose -f local.compose.yml run --rm server -mode=provision -name=me
```

That prints an API key. With it:

```bash
export BOUNDFLOW_API_KEY=<the key it printed>
export BOUNDFLOW_SERVER_ADDRESS=http://localhost:50051
export BOUNDFLOW_WORKER_ADDRESS=http://localhost:50052
export CHARTER_STORE_URL=postgres://charter:charter@localhost:5434/charter
export ANTHROPIC_API_KEY=<your model key>   # or another provider's, see worker.yaml below
```

Remove it with `docker compose -f local.compose.yml down -v`. Cloning the repo
works too, and gets you the examples alongside it.

For production you have two options. Run the BoundFlow backend yourself, following
its [deployment docs](https://github.com/boundflow/boundflow/blob/main/docs/deployment.md).
Or use **BoundFlow Cloud**, which is managed and in early access
([request access](mailto:hello@boundflow.dev)): it gives you an API key and the two
addresses, and you export those instead of the local ones. The worker still runs
wherever you put it, so `CHARTER_STORE_URL` stays yours.

Either way the control plane never sees your model key or its traffic.

### Your first agent

```bash
charter init triage
```

That writes two files. `triage/v1.yaml`, the agent:

```yaml
apiVersion: charter/v1
kind: AgentConfig

name: triage
version: 1
model: claude-haiku-4-5

objective: |
  Triage this support ticket and say what should happen to it:

  {{ inputs.ticket }}

inputs:
  ticket: { type: string, required: true }

response_format:
  category:
    type: string
    description: billing, bug, account, or other.
  next_step:
    type: string
    description: What a person should do about it, in one sentence.
```

and `worker.yaml` beside it, the deployment:

```yaml
apiVersion: charter/v1
kind: Worker

control_plane:
  endpoint: ${BOUNDFLOW_SERVER_ADDRESS}
  worker_endpoint: ${BOUNDFLOW_WORKER_ADDRESS}
  api_key: ${BOUNDFLOW_API_KEY}
  tenant: default

llm:
  # Any provider LangChain can build. Name it here, put its key in the variable
  # below, and install its package: pip install 'boundflow-charter[openai]'
  provider: anthropic
  api_key: ${ANTHROPIC_API_KEY}

store:
  url: ${CHARTER_STORE_URL}

agents_dir: ./
serves:
  - agent: triage
    versions: [1]
```

It calls no tools and sets no budget. Both are optional, and the sections below
add them.

### Run it

```bash
charter tenant create default        # once per control plane
charter agent create triage          # prints an instance id
charter apply .                      # arm config and policy
charter worker .                     # leave this running, it is the process
```

Then, from another terminal, `charter agents` confirms it exists and carries the
instance id every command needs:

When a tool argument belongs to the caller's scope, bind it to a declared agent
input instead of asking the model to repeat it on every call:

```yaml
tool_input_bindings:
  workspace__read_project_file:
    project_id: project_id
```

The outer key is a declared tool name; each mapping pairs a tool argument with
an agent input name. Bound arguments are removed from the model-facing schema
and supplied when the tool executes, including specialist calls. Extra model
arguments cannot override the bound value. Missing inputs or unknown tool
arguments fail before execution. `initial_message` may separately template the
user's request, while `objective` retains stable agent instructions.

For coordinator/specialist workflows, `coordinator_tools` can list the declared
tools exposed to the parent. Omitting it preserves the default of all declared
tools. Specialists still receive their own `tools` lists, with the same caller
bindings and governance wrappers. This separates file creation or other specialist
responsibilities from the parent's inspection and integration work.

For a workflow that requires evidence lookup before delegation, set
`subagent_required_tools: [workspace__read_reference_file]` to declared tools
available to the coordinator. This opt-in middleware refuses `task` until each
named tool has a correlated successful result in the current invocation's history
or its checkpointed tool-completion records.
Pending calls, prose claims and error results do not satisfy it. A lookup and
delegation in the same parallel batch require a subsequent delegation attempt.
Completed tool receipts survive message compaction and checkpoint resume in private
graph state. They cannot be supplied through the agent's public input schema or
borrowed from another conversation. Summary prose never creates a receipt.
This checks tool history, not the relevance or accuracy of retrieved evidence,
and does not replace authorization or independent acceptance tests.

For a coordinator that should delegate initial implementation diagnosis, optional
`coordinator_after_handoff_tools` defers selected declared inspection tools until
its first specialist task returns. Catalog, specification, reference and verification
tools can remain available for choosing a bounded assignment. Deferred tools are
hidden and forced premature calls are rejected; a pending or prerequisite-rejected
handoff does not unlock them. Specialists retain their own configured tools. The
deferred set cannot include prerequisite tools. This is workflow routing, not a
security boundary or proof of successful implementation. After the specialist
returns, the coordinator can inspect source and must still verify integration.

Keep the built-in `read_file` available when using harness compaction/offloading:
large tool results and conversation history are stored in the harness backend.
Use `builtin_tool_hints.read_file` to distinguish these exact temporary artifact
paths from persistent project paths served by custom tools. Scratch writers can
remain excluded while stored results stay readable.

`subagent_review_tools: [workspace__run_acceptance_profile]` additionally requires
each named tool to complete before the first delegation and after the latest
completed specialist batch, including a specialist failure that may have left
partial work. A successfully executed check reporting failed
assertions satisfies the prerequisite for repair; a failed tool invocation does
not. Results must already have completed, so parallel pending calls cannot
stand in for inspected evidence. This opt-in workflow suits agents with applicable
operator checks and usable baseline infrastructure. It does not prove the chosen
check covers a change or that the model understood its result.

`subagent_required_read_tools: [workspace__read_acceptance_result]` requires each
inline specialist to complete the configured diagnostic reads before any write
tool named by `subagent_file_ownership`. Reads must be available to every specialist.
While any prerequisite is pending, those write tools are omitted from the model's
tool list and the required readers receive an explanatory hint. The execution
guard still rejects forced or parallel premature writes. Other read tools remain
available; completing the reads restores the original write tools and descriptions.
Only correlated successful tool results in that specialist's retained history
or checkpointed completion records
count; parent claims, pending calls, execution errors and other conversations do
not. Compaction preserves completion records. This is opt-in and does not inspect
result semantics: an unavailable or irrelevant report can still be a successful
tool invocation. The agent must check availability, freshness and exact failures;
independent verification remains necessary. Only declared write tools are covered.

For dedicated file tools, optional `subagent_file_ownership` adds a required
`owned_files` array to the coordinator's `task` schema:

```yaml
subagent_file_ownership:
  max_files: 4
  write_tools:
    workspace__write_project_file: [path, file_path]
    workspace__patch_project_file: [path, file_path]
    workspace__reuse_design_asset: [destination]
```

Each configured inline specialist can use those tools only for its exact assigned
relative paths. Include every path argument alias. An empty array means read-only.
Overlapping active assignments are refused; success, failure and cancellation
release ownership. The assignment is preserved in the tool call for replay and
passed into the child invocation through context-local state, separately for each
concurrent task. Precompiled specialists and unconfigured default specialists
cannot participate in this mode. Paths use local platform case normalization.
This coordinates listed tools within one parent run; it does not constrain
unlisted tools, shell side effects, symlink aliases or other runs. Keep separate
filesystem and command controls, and bind every task to the intended project.

Use `subagent_context_inputs: [project_id, brief]` to preserve selected declared
caller inputs verbatim in every inline `task` description. This prevents a
coordinator summary from being the specialist's only view of the request. The
same selected inputs are supplied as a user-role reminder once the exact copy
in the original handoff is no longer present, including after context
summarization. A generated summary or tool result cannot suppress that reminder.
This avoids duplicating the handoff context in each initial model request and
does not append repeated reminders to saved history. The default is empty; unselected inputs
are not added. Select only context suitable
for every specialist, never credentials. The assigned scope, ownership and tool
permissions still apply. This mechanism transports context; it does not prove
that a model followed it.

Use `coordinator_context_inputs: [project_id, user_request]` when the parent also
needs selected original inputs visible after compaction. These declared inputs
are reattached as a user-role reminder to each coordinator model request, without
accumulating in saved history. The reminder distinguishes original constraints
from generated diagnoses and does not grant permissions. It is opt-in and must
not include credentials. Known model context limits reserve estimated space for
the larger parent/specialist reminder in addition to maximum output, leaving
less space for compactable history. Oversized reservations fail explicitly;
unknown provider limits are not invented. Exact transport does not establish
semantic compliance, so independent checks remain necessary.

Agents using dedicated project tools can set
`builtin_tools_exclude: [write_file, edit_file]` to hide competing scratch writes
from the parent and inline specialists and reject calls to those names. The default
is unchanged. Keep `read_file` available when the harness needs to retrieve saved
history or offloaded tool output. This controls tool routing, not OS access.
`builtin_tool_hints` optionally maps built-in names to routing guidance, which is
prepended to their model-facing descriptions and appended to error responses.
It does not change tool permissions or make scratch files into project files.

For action-oriented agents, `tool_format_retries: 2` can recover when a model
prints `<function=known_tool>` markup instead of a structured tool call. The default
is disabled; the configured maximum is three retries per malformed response.
The parent and inline specialists receive a correction asking for native tool use.
By default printed text is never executed. Repeated malformed output fails the run, and retries
still use the governed model and its call/token limits. This is not a general parser
for arbitrary tool syntax or proof that a completed response fulfilled the task.

For observed read loops, `repeated_read_tools: [workspace__read_project_file]`
opts declared read-only tools into recovery. Three consecutive reads or batches
with identical arguments and successful text results add computed feedback to the
next model request. Parallel result order does not affect matching. Six unchanged
reads or batches fail the run visibly. Changed arguments,
changed results, errors, intervening actions, and new user messages reset detection.
Parent and inline specialists receive the same protection; default behavior is
unchanged. Results and saved history are preserved, no action is synthesized, and
ownership, prerequisites and budgets still apply. This detects one narrow form of
stalled work; it does not establish semantic progress or task completion.

For Qwen-coder providers that omit the opening `<tool_call>` tag,
`normalize_qwen_tool_fragments: true` opts into a narrow compatibility parser.
Only an entire assistant response containing one closed native function fragment,
with an advertised tool name and schema-valid arguments, becomes a structured call.
Prose, mixed calls, duplicate/unknown fields and incomplete fragments are rejected.
The ordinary governed tool loop still authorizes and executes the call; the parser
never executes anything. The original provider response remains in the trace, and
the normalized message records the parser name and original text hash. This option
is disabled by default and is separate from retrying malformed responses.

```
AGENT   INSTANCE  VER  STATUS  ACTIVITY
triage  d9811374  v1   active  active
```

Run a task against it:

```bash
charter run triage --instance d9811374 --ticket "card declined twice, tried a new one"  # prints a task id
charter status <task-id>
```

`status` prints what the agent returned, in the shape `response_format` declared:

```
task      f683f822-d8f4-40a7-b528-8db1a576140c
outcome   successful
took      11s

inputs
  ticket   card declined twice, tried a new one

result
  category    billing
  next_step   Verify if the new card payment processed successfully and contact
              the customer to resolve any ongoing payment issues.
```

The console shows the same thing in a browser, for all agents:

```bash
charter ui
```

### A local model

Inference is bring-your-own, so a model on your own machine is a provider like any
other. With [Ollama](https://ollama.com), `pip install 'boundflow-charter[ollama]'`
and point `worker.yaml` at it:

```yaml
llm:
  provider: ollama        # no api_key: a local runtime needs none
```

Name the model in the agent's version file as usual — `model: qwen2.5:7b`. For
more information see [docs/local-models.md](docs/local-models.md).

## Approvals and policy

Tools can be gated on human approval. Behaviour is versioned, so adding one means
writing a new version file:

```yaml
mcp:
  - name: stripe
    url: https://mcp.stripe.com
    env: [STRIPE_API_KEY]
    tools:
      - tool: get_charge
      - tool: create_refund
        approval: always
```

Charter stops the task and shows a person the call it wants to make and the
reasoning behind it:

```bash
charter approve apr_01J8Z --actor dana --reason "third dispute this month"
```

Nothing waits in your terminal. The task ends at the gate and resumes when someone
answers, which can be days later on a different worker.

Limits are policy rather than behaviour, so they sit outside the version.
`runtime.yaml` holds what one task may spend and what the agent may reach:

```yaml
apiVersion: charter/v1
kind: RuntimePolicy
agent: triage

per_run:
  max_cost_usd: 0.50
  max_llm_calls: 20
  max_seconds: 300
  max_parallel_subagents: 3
  capability_call_limits:
    - { capability: write, max_calls: 10 }

limits:
  max_call_seconds: 60
  max_tool_seconds: 30

authority:
  allowed_capabilities: [read, write]
  approval_timeout_seconds: 3600
```

Operation deadlines are derived from the call and run budgets, with a one-hour
ceiling by default. For an explicitly longer task, set
`operation_timeout_ceiling_seconds: 7200` at the top level of `runtime.yaml` and
give `per_run.max_seconds` enough time. The configurable ceiling accepts 60 through
86,400 seconds; it does not raise call or run budgets. Entry operations and later
rounds use the same derived deadline. Apply the policy before starting the task;
changing it does not extend an already dispatched operation.

When `subagent_review_tools` is configured, delegation also carries bounded
excerpts of the latest successful review-tool executions into the specialist's
assignment. Failed assertions remain visible; invocation errors do not qualify.
The excerpts preserve exact diagnostic values separately from the coordinator's
interpretation, disclose truncation, and remain untrusted data rather than new
instructions or permissions.

`lifecycle.yaml` acts on the agent over time. When a metric crosses a threshold the
control plane can pause it, cool it down, or roll it back to an earlier version:

```yaml
apiVersion: charter/v1
kind: LifecyclePolicy
agent: triage

rules:
  - when: { metric: num_failures, threshold: 3 }
    then: { pause: { window: 5 } }

  - when: { metric: approval_rejections, threshold: 2 }
    then: { cooldown: { window: 10, seconds: 3600 } }

  - when: { metric: cost, threshold: 2.00 }
    then: { set_version: { target: 1 } }
```

Both are re-applied on every `charter apply`, so a ceiling can be lowered without
cutting a release.

## Architecture

`charter apply` compiles your configuration into workflows and policy on the
[BoundFlow](https://github.com/boundflow/boundflow) control plane. A Charter worker
runs the agent in your environment and talks to your MCP servers with credentials
that stay there. Each worker registers the agents it can run, listed under `serves`
in its `worker.yaml`, and any worker registered for an agent can pick up its work.
If the worker running a task crashes or stops, another continues it from its last
checkpoint, and a run parked for a person resumes on whichever worker is free when
they answer.

The agent loop itself is [deepagents](https://github.com/langchain-ai/deepagents),
so its tools, subagents, filesystem and skills work here unchanged. Charter makes
that loop durable and governed: it checkpoints the run, turns the harness's
interrupts into approvals a person can answer tomorrow, and holds it to the limits
your config declares.

```
                     BoundFlow
                   Control Plane
            state • policy • lifecycle
                         │
                        RPC
             ┌───────────┴───────────┐
             ▼                       ▼
  ┌────────────────────┐  ┌────────────────────┐
  │ Charter worker     │  │ Charter worker     │
  │                    │  │                    │
  │ model ↔ agent loop │  │ model ↔ agent loop │
  │          │         │  │          │         │
  │      MCP tools     │  │      MCP tools     │
  └────────────────────┘  └────────────────────┘
                 Your environment
```

Charter adds no database or service of its own. Deployed agents keep running
through their workers and the control plane whether or not the CLI is installed.

## Documentation

- [DESIGN.md](DESIGN.md): every field of every file, and the decisions behind them
- [docs/local-models.md](docs/local-models.md): running an agent on a model on your
  own machine, and what differs from a hosted provider
- [deploy/](deploy/): running workers as containers, and a control plane locally
- [examples/](examples/): two agents over a toy support system. One gates a refund
  and pauses itself when too many are turned down, the other rolls itself back to
  an earlier version. They run with nothing but a model key

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,ui,otel]'
.venv/bin/pytest
```

`boundflow` comes from PyPI. Add `--pre --upgrade boundflow` to track its main,
which is what CI's second unit job does.

End-to-end tests need a control plane, and skip themselves without one. The compose
file CI uses runs the published image:

```bash
docker compose -f deploy/local.compose.yml up -d --wait
key=$(docker compose -f deploy/local.compose.yml run --rm server \
        -mode=provision -name=dev | awk '/^api_key/{print $NF}')

export BOUNDFLOW_API_KEY=$key
export BOUNDFLOW_SERVER_ADDRESS=http://localhost:50051
export BOUNDFLOW_WORKER_ADDRESS=http://localhost:50052
export CHARTER_STORE_URL=postgres://charter:charter@localhost:5434/charter
pytest tests/e2e
```

They use a real control plane, a real MCP subprocess and real governance gates.
Only the model is faked, so the suite stays deterministic and free.
