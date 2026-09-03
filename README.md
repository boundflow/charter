# Charter

**Build and manage production-ready agents that run on your own compute.**

![The Charter console: the fleet with an agent parked on an approval, the decision waiting on a human, and the policy and run history behind it](docs/console.gif)

> **Pre-alpha, and in the open early.** The design is settled enough to read and
> argue with. The code is not settled enough to run anything you care about.
> Expect the configuration format to change.

Charter provides the infrastructure for running AI agents in production. You define
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
- **Your network, your data.** Workers run in your environment, so agents reach
  internal services and databases directly. Model keys and prompts never reach the
  control plane, and the agent's conversation, files and traces stay in stores you
  run.

[DESIGN.md](DESIGN.md) documents every field.

## Quickstart

```bash
pip install boundflow-charter          # add [ui] for the console, [otel] for traces
pip install --pre boundflow-charter    # or whatever main is, published every green build
```

### A control plane

Charter needs one to run agents against. To run one locally:

```bash
docker compose -f deploy/local.compose.yml up -d --wait
docker compose -f deploy/local.compose.yml run --rm server -mode=provision -name=me
```

That prints an API key. With it:

```bash
export BOUNDFLOW_API_KEY=<the key it printed>
export BOUNDFLOW_SERVER_ADDRESS=http://localhost:50051
export BOUNDFLOW_WORKER_ADDRESS=http://localhost:50052
export CHARTER_STORE_URL=postgres://charter:charter@localhost:5434/charter
```

Remove it with `docker compose -f deploy/local.compose.yml down -v`.

For production you have two options. Run the BoundFlow backend yourself, following
its [deployment docs](https://github.com/boundflow/boundflow/blob/main/docs/deployment.md).
Or use **BoundFlow Cloud**, which is managed and in early access
([request access](mailto:hello@boundflow.dev)): it gives you an API key and the two
addresses, and you export those instead of the local ones. The worker still runs
wherever you put it, so `CHARTER_STORE_URL` stays yours.

Either way the control plane never sees your model key or its traffic.

### Your first agent

`summarize/v1.yaml`:

```yaml
apiVersion: charter/v1
kind: AgentConfig

name: summarize
version: 1
model: claude-haiku-4-5

objective: |
  Summarise this in two sentences: {{ inputs.text }}

inputs:
  text: { type: string, required: true }

response_format:
  summary:
    type: string
    description: The summary, in two sentences.
```

`worker.yaml`, beside it:

```yaml
apiVersion: charter/v1
kind: Worker

control_plane:
  endpoint: ${BOUNDFLOW_SERVER_ADDRESS}
  worker_endpoint: ${BOUNDFLOW_WORKER_ADDRESS}
  api_key: ${BOUNDFLOW_API_KEY}
  tenant: default

llm:
  provider: anthropic
  api_key: ${ANTHROPIC_API_KEY}

store:
  url: ${CHARTER_STORE_URL}

agents_dir: ./
serves:
  - agent: summarize
    versions: [1]
```

It calls no tools and sets no budget. Both are optional, and the sections below
add them.

### Run it

```bash
charter tenant create default        # once per control plane
charter agent create summarize       # prints an instance id
charter apply .                      # arm config and policy
charter worker .                     # leave this running, it is the process
```

Then, from another terminal:

```bash
charter run summarize --instance <id> --text "..."
charter status <task-id>
```

## How it works

A tool the agent shouldn't call on its own gets one line:

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

When the agent decides to call `create_refund`, Charter stops the task and shows a
human the call it wants to make and the reasoning behind it:

```bash
charter approve apr_01J8Z --reason "third dispute this month"
```

The refund runs after you approve it. The agent gets the result, finishes the task,
and reports what it did.

Nothing waited in your terminal for that. `charter apply` compiles your
configuration into workflows and policy on the
[BoundFlow](https://github.com/boundflow/boundflow) control plane. A Charter worker
runs the agent in your environment and talks to your MCP servers with credentials
that stay there.

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

Charter adds no database or service of its own. Deployed agents keep running
through their workers and the control plane whether or not the CLI is installed.

## Documentation

- [DESIGN.md](DESIGN.md): every field of every file, and the decisions behind them
- [deploy/](deploy/): running workers as containers, and a control plane locally
- [examples/](examples/): fuller configurations, for reading. They name real
  Zendesk and Stripe servers, so they do not run as-is
- [demo/leads/](demo/leads/): an agent that runs end to end against a local MCP
  server, where you play the people it contacts

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
