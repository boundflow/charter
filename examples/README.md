# Examples

Two agents over one toy support system. `support_server.py` is a small MCP server
with four tickets and the charges behind them, so both run with nothing but a
model key.

    refund-triage       reads a ticket, decides, and refunds. The refund stops
                        for a human, and the agent pauses itself if too many are
                        turned down.
    ticket-summarizer   reads every open ticket and reports what needs attention.
                        Runs unattended on v2, and rolls itself back to v1 when
                        it spends too much.

## Running them

You need a control plane and the environment from the
[Quickstart](../README.md#quickstart), plus `ANTHROPIC_API_KEY`.

Start the support server. It serves MCP over HTTP on port 8931, and both agents
reach it by URL:

    python examples/support_server.py

In a second terminal, from this directory, bring up an agent and a worker:

    charter tenant create default        # once per control plane
    charter agent create refund-triage
    charter apply .
    charter worker .

`charter ui` opens the console, where the agent appears as soon as it exists.
Leave it open: everything below shows up there as it happens.

In a third, give it a ticket:

    charter run refund-triage --instance <id> --ticket_id T-1041

Tickets are T-1041, T-1042, T-1043 and T-1044.

Nothing holds your terminal open while it runs. In the console the task appears,
then a gate when the agent proposes a refund. Answer it there, with a box for
your reason.

From a terminal instead:

    charter pending refund-triage --instance <id>

That prints the call it wants to make and the two commands that answer it. Both
take a reason, and the reason is not paperwork: it is handed to the agent.

    charter approve <id> --agent refund-triage --instance <id> --reason '...'
    charter reject  <id> --agent refund-triage --instance <id> --reason '...'

Approve, and the refund goes through and the task finishes with what it did. The
console shows it pick back up, and the audit there records who decided and why.

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

The `runtime.yaml` we applied lets one refund go through per task, and lets the
agent propose twice. A rejected proposal never runs, so it costs nothing against
the first number. The second is what stops it asking you a third time.

The `lifecycle.yaml` we applied pauses the agent after three rejections across the
last three runs. Keep rejecting, on this ticket or another, until it stops:

    charter describe refund-triage --instance <id>

    refund-triage
      version    v1
      status     paused
      activity   active

If the third rejection lands mid-run, the agent stays active until that run
finishes. Lifecycle rules are evaluated between runs, not during one, so they
decide whether the next run starts rather than stopping the one in flight.

Further runs are refused until you resume it:

    charter resume refund-triage --instance <id>

## Rolling a version back

v2 is a new version of `ticket-summarizer`'s prompt. If it turns out to cost too
much, roll back to v1:

    - when: { metric: cost, threshold: 0.05 }
      then: { set_version: { target: 1 } }

Create it and let it run:

    charter agent create ticket-summarizer
    charter apply . --all
    charter run ticket-summarizer --instance <id>

It starts on v2. Once v2 has spent five cents the control plane puts v1 back,
with nobody watching:

    AGENT              INSTANCE  VER  STATUS  ACTIVITY
    ticket-summarizer  b4a3491a  v1   active  active

    charter audit ticket-summarizer --instance b4a3491a
    2026-09-08 22:12  policy fired: metric=cost action=set_version

`charter describe` shows the rules and what the running version has spent so far.

## What each file is for

`v1.yaml` is behaviour, and it is versioned: you write a v2 rather than editing
it. `runtime.yaml` is what one task may spend and what the agent may reach.
`lifecycle.yaml` is what the control plane does to the agent between tasks.
Those last two are re-applied on every `charter apply`, so a ceiling can be
lowered without cutting a release.

`worker.yaml` is the deployment: which control plane, whose credentials, which
agents this process serves. The notification and tracing blocks are commented
out, because uncommented they need the environment variables they name.
