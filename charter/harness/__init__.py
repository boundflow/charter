"""Making deepagents durable and governed.

This is where Charter's choice of harness is spent. It opens the agent's
checkpointer and store against Postgres, keyed so a task resumes on the thread it
parked on; it turns the harness's interrupts into BoundFlow's gates and back; and
it holds the harness to the policy the agent's config declared.

The division with BoundFlow: it owns the governor that counts spend and enforces
ceilings, the policy types that travel over the wire and are stored and versioned
server-side, and the LangChain client. Those are harness-agnostic on purpose — a
control plane governs whatever loop you bring it. Everything that knows the loop is
deepagents lives here, and talks to that half through its public surface:
`agent_governor`, `run_governed`, `workflow_id`, `request_id`.

    async with durable_harness(ctx, agent_name, store_url) as h:
        result = await ctx.run_governed(agent_name, lambda model, tools: ...)
"""

from .durable import DurableHarness, durable_harness, task_context
from .gates import approve, pending_action, reject, respond

__all__ = [
    "DurableHarness",
    "approve",
    "durable_harness",
    "pending_action",
    "reject",
    "respond",
    "task_context",
]
