# Examples

Two agents over one toy support desk. `desk.py` is a small MCP server with four
tickets and the charges behind them, so both run with nothing but a model key.

    refund-triage       reads a ticket, decides, and refunds. The refund stops
                        for a human before it goes through.
    ticket-summarizer   reads every open ticket and reports what needs attention.
                        Two versions, so a rollback has somewhere to go.

## Running them

You need a control plane and the environment from the
[Quickstart](../README.md#quickstart), plus `ANTHROPIC_API_KEY`.

Start the desk. It serves MCP over HTTP on port 8931, and both agents reach it by
URL:

    python examples/desk.py

In a second terminal, from this directory, bring up an agent and a worker:

    charter agent create refund-triage
    charter apply .
    charter worker .

In a third, give it a ticket:

    charter run refund-triage --instance <id> --ticket_id T-1041

The agent reads the ticket, looks up the charge, and asks to refund it. Nothing
holds your terminal open while it waits:

    charter pending refund-triage --instance <id>

That prints the call it wants to make and the commands that answer it. Approve,
and the refund goes through and the task finishes. Reject, and the agent is told
and carries on without it.

`charter ui` shows the same thing in a browser, across every agent at once.

The desk has four tickets, T-1041 to T-1044. One is a duplicate charge, one is a
size exchange the refund policy says not to refund.

## What each file is for

`v1.yaml` is behaviour, and it is versioned: you write a v2 rather than editing
it. `runtime.yaml` is what one task may spend and what the agent may reach.
`lifecycle.yaml` is what the control plane does to the agent between tasks.
Those last two are re-applied on every `charter apply`, so a ceiling can be
lowered without cutting a release.

`worker.yaml` is the deployment: which control plane, whose credentials, which
agents this process serves. The notification and tracing blocks are commented
out, because uncommented they need the environment variables they name.
