"""One operation. The harness does the rest.

    entry -> run the harness -> Complete
                             \\-> AwaitApproval -> entry (with the decision)

Charter used to drive the agent: rounds, drafts, questions, a propose/execute
split, and a history it reassembled into every prompt. All of that was a loop, and
the harness has a better one. What's left here is the part a harness structurally
cannot do — park a run for a human, survive the worker dying, resume somewhere
else, and stop when the money runs out.

The gate is the whole trick. deepagents decides *what* needs a human, through a
tool's `interrupt_on` or a `file_rules` interrupt, and raises a LangGraph interrupt
to say so. That interrupt is an exception: it unwinds the graph and dies with the
process. We catch it at the top, turn it into an `AwaitApproval`, and hand the
decision back on the far side as a `Command(resume=...)`. Same moment the harness
chose; a pause that outlives the machine that started it.
"""

from __future__ import annotations

import logging
from typing import Any

from boundflow import (
    ENTRY_OPERATION,
    AwaitApproval,
    Budget,
    Complete,
    Next,
)
from boundflow.harness import durable_harness, task_context
from boundflow.harness_gates import approve, pending_action, reject
from boundflow.llm import AgentPolicyLimitExceeded

from ..config.agent import AgentConfig

log = logging.getLogger(__name__)

# Context keys Charter owns. Underscored so they can't collide with an author's
# declared inputs, which share the same dict.
K_DECISION = "_decision"
K_COST = "_cost"
K_LLM_CALLS = "_llm_calls"
K_GATES = "_gates"


def render(template: str, inputs: dict) -> str:
    """`{{ inputs.<name> }}` and nothing else. Validated at apply time, so anything
    unresolved here is a missing invoke context rather than a bad template."""
    out = template
    for name, value in inputs.items():
        for spelling in (f"{{{{ inputs.{name} }}}}", f"{{{{inputs.{name}}}}}"):
            out = out.replace(spelling, str(value))
    return out


def interrupt_on(cfg: AgentConfig) -> dict[str, Any]:
    """Which tools stop for a human, in the harness's own vocabulary.

    `edit` is offered wherever the harness allows it: an approver correcting an
    amount is a better outcome than rejecting and hoping the next draft is right,
    and it costs nothing to permit — a decision still has to be made either way.
    """
    return {tool: {"allowed_decisions": ["approve", "edit", "reject"]}
            for tool in cfg.gated_tools}


def response_schema(cfg: AgentConfig) -> dict | None:
    """The agent's structured answer, or None to let it reply in prose.

    Charter adds no fields of its own any more. What comes back is the agent's
    result, not a wrapper we have to unpick.
    """
    if not cfg.response_format:
        return None
    return {
        "type": "object",
        "properties": {name: {"type": spec.type,
                              **({"description": spec.description} if spec.description else {})}
                       for name, spec in cfg.response_format.items()},
        "required": list(cfg.response_format),
        "additionalProperties": False,
    }


class Ended(Exception):
    """A per-task budget is spent. Carries what to tell the operator."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class Loop:
    """One agent version's handler. Holds no per-task state — everything lives in
    `ctx.context` and the harness's own checkpoint, so a task can resume on a
    different worker after a gate."""

    def __init__(self, cfg: AgentConfig, runtime, tools, chat_model,
                 store_url: str) -> None:
        self.cfg = cfg
        self.runtime = runtime
        # The ToolSet, not its tools: it hasn't connected yet when the worker builds
        # this, and a quarantined agent reconnects later. Asked per task instead.
        self.tools = tools
        self.chat_model = chat_model  # a factory: (model_name) -> BaseChatModel
        self.store_url = store_url

    # ── the budget the harness can't see ────────────────────────────────────

    def _operation_timeout(self) -> int:
        return self.runtime.operation_timeout_seconds

    def _remaining(self, ctx) -> Budget | None:
        """What's left of the task budget, for this one stretch of work.

        Still needed, and for the same reason as before: a gate ends the operation,
        and the next one gets a fresh runtime policy. Without this a task that
        stopped for a human four times could spend its declared budget five times
        over. `Budget` can only tighten, and refuses at zero — which matters,
        because 0 means "unlimited" in RuntimePolicy, so passing it through would
        lift the cap at the exact moment it should bite.
        """
        per_run = self.runtime.per_run
        c = ctx.context
        fields: dict[str, Any] = {}
        if per_run.max_cost_usd:
            fields["max_cost_usd"] = per_run.max_cost_usd - c.get(K_COST, 0.0)
        if per_run.max_llm_calls:
            fields["max_llm_calls"] = per_run.max_llm_calls - c.get(K_LLM_CALLS, 0)

        # Every declared tool gets the same failure allowance. No harness equivalent
        # exists, so a broken integration would otherwise burn the budget rather than
        # trip its own breaker.
        if per_run.max_tool_failures:
            fields["tool_failure_limits"] = {
                tool: per_run.max_tool_failures for tool in self.cfg.all_tools}

        return Budget(**fields) if fields else None

    def _charge(self, ctx, result) -> None:
        c = ctx.context
        c[K_COST] = c.get(K_COST, 0.0) + result.cost_usd
        c[K_LLM_CALLS] = c.get(K_LLM_CALLS, 0) + result.llm_calls_used

    def _fail(self, ctx, reason: str) -> Complete:
        """Record a customer-domain failure and still return a result.

        `mark_failed()` increments num_failures for the lifecycle rules while the run
        completes normally, so the task reports how far it got instead of dying with
        a stack trace — which matters, because this payload is what an operator reads
        when a rule pauses the agent.
        """
        ctx.mark_failed()
        log.warning("task failed: agent=%s reason=%s", self.cfg.name, reason)
        return Complete(result={
            "failed": True,
            "reason": reason,
            "cost_usd": round(ctx.context.get(K_COST, 0.0), 6),
            "llm_calls": ctx.context.get(K_LLM_CALLS, 0),
            "gates": ctx.context.get(K_GATES, 0),
        })

    # ── the one operation ───────────────────────────────────────────────────

    async def entry(self, ctx):
        """Run the harness until it finishes or asks for a human.

        Serves both a fresh task and every resume after a gate. `h.first()` decides
        which — a payload or the parked interrupt's resume command — so there's no
        branch here that could get the two out of step.
        """
        from deepagents import create_deep_agent

        # Built here rather than when the gate was raised, because the approver's
        # reason doesn't exist yet at that point — and a rejection the agent can't
        # read the reason for is the least useful kind. A timeout lands on the same
        # branch with no decider, so it arrives as a rejection without one.
        verdict = ctx.context.pop(K_DECISION, None)
        decision = None
        if verdict == "approve":
            decision = approve()
        elif verdict == "reject":
            decision = reject(ctx.approval_reason or "no reason given")

        try:
            async with durable_harness(ctx, self.cfg.name, self.store_url,
                                       resume=decision) as h:
                prompt = render(self.cfg.objective, ctx.context)
                result = await ctx.run_governed(
                    self.cfg.name,
                    lambda model, tools: create_deep_agent(
                        model=model,
                        tools=tools,
                        system_prompt=prompt,
                        # Ours to declare, theirs to enforce, durable because of us.
                        interrupt_on=interrupt_on(self.cfg),
                        **h.wiring,
                    ).ainvoke(h.first({"messages": [{"role": "user", "content": prompt}]}),
                              h.config),
                    chat_model=self.chat_model(self.cfg.model),
                    tools=self.tools.langchain_tools(),
                    output_schema=response_schema(self.cfg),
                    budget=self._remaining(ctx),
                )
        except AgentPolicyLimitExceeded as spent:
            return self._fail(ctx, str(spent))
        except Ended as ended:
            return self._fail(ctx, ended.reason)

        self._charge(ctx, result)

        if (action := pending_action(result)) is not None:
            return self._gate(ctx, action)
        return Complete(result=result.output)

    def _gate(self, ctx, action: dict):
        """Park for a human on the action the harness stopped at.

        Only the first pending action is presented even when a turn proposes
        several: a workflow holds one gate at a time, so the rest are decided on
        subsequent rounds rather than lost.
        """
        c = ctx.context
        c[K_GATES] = c.get(K_GATES, 0) + 1
        gate = self.cfg.gate

        def resume(verdict: str):
            return Next(ENTRY_OPERATION,
                        context=task_context(ctx, {**c, K_DECISION: verdict}),
                        timeout=self._operation_timeout())

        log.info("gate: agent=%s tool=%s", self.cfg.name, action.get("name", "?"))
        return AwaitApproval(
            on_approve=resume("approve"),
            on_reject=resume("reject"),
            timeout=gate.timeout_seconds,
            justification=self._justify(action),
            metadata={"tool": action.get("name", ""), "args": action.get("args", {})},
        )

    def _justify(self, action: dict) -> str:
        """What the approver reads. The harness already writes a display string for
        its own in-process prompt; we use it rather than composing a second one that
        could describe the same action differently."""
        return (action.get("description")
                or f"{self.cfg.name} wants to call {action.get('name', 'a tool')}")


def operations(loop: Loop) -> dict:
    """Every operation this workflow has. One, now."""
    return {ENTRY_OPERATION: loop.entry}
