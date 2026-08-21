"""Turn a harness's in-process interrupt into a durable BoundFlow gate.

deepagents already decides *what* needs a human: `permissions` with `mode="interrupt"`,
or `interrupt_on={"tool": True}` for any tool at all. What it can't do is *wait*. Its
`HumanInTheLoopMiddleware` raises a LangGraph interrupt, which means the process holds
the pause — close the laptop and it's gone.

That's the seam. The harness names the moment; BoundFlow owns the waiting. Park the
operation, let the worker die, approve two days later, resume on a different machine.

    result = await ctx.run_governed(...)

    if (action := pending_action(result)) is not None:
        return AwaitApproval(
            on_approve=Next("resume", context={"decision": approve()}, timeout=300),
            on_reject=Next("resume", context={"decision": reject()}, timeout=300),
            justification=action["description"],
            timeout=86_400)

and on the far side, feed the decision back with `Command(resume=...)`:

    agent.ainvoke(Command(resume=ctx.context["decision"]), {"configurable": {...}})

Resuming needs the same `thread_id` and a checkpointer both operations can reach — the
interrupt lives in the graph's state, not in the process that raised it.
"""
from __future__ import annotations

from typing import Any


def pending_action(result: Any) -> dict | None:
    """The action a human has to decide on, or None if the harness just finished.

    Accepts the `StepResult` from `run_governed` or the raw state dict, since callers
    reasonably have either. Returns the first pending request: name, args and a
    ready-to-display description.

    Only the first is returned even when a turn proposes several. A workflow holds one
    gate at a time (`jobs.workflow_id` is the primary key), so the rest are decided on
    subsequent rounds rather than lost.
    """
    state = getattr(result, "output", result) or {}
    if not isinstance(state, dict):
        return None
    interrupts = state.get("__interrupt__") or []
    for item in interrupts:
        value = getattr(item, "value", item)
        if not isinstance(value, dict):
            continue
        requests = value.get("action_requests") or []
        if requests:
            action = dict(requests[0])
            action["interrupt_id"] = getattr(item, "id", "")
            return action
    return None


def approve() -> dict:
    """A resume payload approving the pending action as proposed."""
    return {"decisions": [{"type": "approve"}]}


def reject(reason: str = "") -> dict:
    """Reject the action. The harness tells the model, which continues without it —
    a refusal is feedback, not a failed run."""
    decision: dict = {"type": "reject"}
    if reason:
        decision["message"] = reason
    return {"decisions": [decision]}


def respond(message: str) -> dict:
    """Answer the agent instead of deciding on the action — the `AwaitInput` shape.
    Use when the human's reply is information rather than a yes or no."""
    # "respond", not "response". The harness's DecisionType is a closed literal
    # and a wrong spelling doesn't raise — the resume is simply never matched, so
    # the run parks forever and looks like a hang rather than a typo.
    return {"decisions": [{"type": "respond", "message": message}]}


def edit(args: dict) -> dict:
    """Approve the action but with different arguments — the human corrects the agent
    rather than blocking it. Only valid when the harness listed `edit` in the request's
    `allowed_decisions`."""
    return {"decisions": [{"type": "edit", "args": args}]}
