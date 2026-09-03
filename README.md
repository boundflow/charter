# Charter

**The easiest way to build and manage production-ready agents.**

![The Charter console: the fleet with an agent parked on an approval, the decision waiting on a human, and the policy and run history behind it](docs/console.gif)

> **Pre-alpha, and in the open early.** The design is settled enough to read and
> argue with; the code is not settled enough to run anything you care about.
> Expect the configuration format to change.

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

## Why Charter

- **Automatic rollback** — an agent whose failures, spend or rejections cross a
  threshold you set pauses, cools down, or returns to the version that worked.
- **Durable approvals** — a run parks at a gate and resumes days later, on another
  worker, with everything it had discovered, spent and been told.
- **Per-task budgets** — cost, model calls and tool failures are capped across the
  whole task, and an exhausted limit names which one and what it suggests.
- **Declared authority** — the model sees only the tools you listed, and the ones
  you gate it can propose but never call.
- **Versioned behaviour** — objective, tools and gates are an immutable file, so a
  rollback restores the whole agent rather than a prompt string.
- **Traces you own** — every model and tool call to your own OTLP backend; prompts
  never reach the control plane.

Each is a few lines of YAML. [DESIGN.md](DESIGN.md) is the field reference.

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

For production, either run the BoundFlow backend yourself — its
[deployment docs](https://github.com/boundflow/boundflow/blob/main/docs/deployment.md)
own that — or use **BoundFlow Cloud**, managed and in early access
([request access](mailto:hello@boundflow.dev)). Cloud hands you an API key and the
two addresses; export those instead of the local ones and nothing else changes.
The worker still runs wherever you put it, so `CHARTER_STORE_URL` is still yours.

Whichever you pick, inference stays yours: the control plane never sees your model
key or its traffic.

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
charter worker .                     # in its own terminal — this is the process
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

Suppose the agent reads a ticket and concludes a $240 refund is warranted. It
cannot issue it. `create_refund` requires approval, so Charter parks the task and
surfaces the proposed call and the agent's reasoning to a human:

```bash
charter approve apr_01J8Z --reason "third dispute this month"
```

Only then is the refund executed. The agent receives the result, finishes the task,
and reports what happened. The task didn't live in your terminal in the meantime —
`charter apply` compiled your configuration into workflows and policy on the
[BoundFlow](https://github.com/boundflow/boundflow) control plane, and a Charter
worker runs the agent loop in your environment, talking to your MCP servers with
credentials that never leave it.

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

## Documentation

- [DESIGN.md](DESIGN.md) — every field of every file, and the decisions behind them
- [demo/leads/](demo/leads/) — an agent that finds people, asks you to approve
  every message before it sends one, and waits days if that is how long you take.
  It runs against a fake network on your machine, so nothing reaches anybody.
- [examples/](examples/) — fuller configurations, for reading. They name real
  Zendesk and Stripe servers, so they do not run as-is.
- [deploy/](deploy/) — running workers as containers, and a control plane locally

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,ui,otel]'
.venv/bin/pytest
```

`boundflow` comes from PyPI; add `--pre --upgrade boundflow` to track its main, which
is what CI's second unit job does.

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
