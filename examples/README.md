# Examples

Two agents over one toy support system. `support_server.py` is a small MCP server
with four tickets and the charges behind them, so both run with nothing but a
model key.

    refund-triage       reads a ticket, decides, and refunds. The refund stops
                        for a human before it goes through.
    ticket-summarizer   reads every open ticket and reports what needs attention.
                        Two versions, so a rollback has somewhere to go.

## Running them

You need a control plane and the environment from the
[Quickstart](../README.md#quickstart), plus `ANTHROPIC_API_KEY`.

Start the support server. It serves MCP over HTTP on port 8931, and both agents
reach it by URL:

    python examples/support_server.py

In a second terminal, from this directory, bring up an agent and a worker:

    charter agent create refund-triage
    charter apply .
    charter worker .

In a third, give it a ticket:

    charter run refund-triage --instance <id> --ticket_id T-1041

The agent reads the ticket, looks up the charge, and asks to refund it. Nothing
holds your terminal open while it waits:

    charter pending refund-triage --instance <id>

That prints the call it wants to make and the two commands that answer it. Both
take a reason, and the reason is not paperwork: it is handed to the agent.

    charter approve <id> --agent refund-triage --instance <id> --reason '...'
    charter reject  <id> --agent refund-triage --instance <id> --reason '...'

Approve, and the refund goes through and the task finishes with what it did.

Reject with a reason that says what was wrong, and the agent works from it:

    charter reject <id> --agent refund-triage --instance <id> \
      --reason 'only half of this is ours. the second charge was authorised by the
                customer on a different order, so refund 24.00, not 48.00'

It comes back with a corrected proposal, and a second gate:

    refund-triage wants to call support__create_refund
      with charge_id='ch_88213', amount_usd=24.0

Approve that one and the task finishes:

    result
      refunded_usd   24.0
      resolution     Customer was charged twice for order #4417 on the 3rd at
                     $48.00 each. Refunded $24.00 for our duplicate charge; the
                     other $48.00 charge was authorized by the customer and remains.

The agent will keep revising while you keep giving it reasons, so `runtime.yaml`
caps `support__create_refund` at three calls per task. The objective asks it to
revise rather than repeat; the ceiling is what holds when it doesn't.

`charter status <task-id>` is where you read the outcome, and `charter ui` does
all of this in a browser, across every agent at once.

## What each file is for

`v1.yaml` is behaviour, and it is versioned: you write a v2 rather than editing
it. `runtime.yaml` is what one task may spend and what the agent may reach.
`lifecycle.yaml` is what the control plane does to the agent between tasks.
Those last two are re-applied on every `charter apply`, so a ceiling can be
lowered without cutting a release.

`worker.yaml` is the deployment: which control plane, whose credentials, which
agents this process serves. The notification and tracing blocks are commented
out, because uncommented they need the environment variables they name.
