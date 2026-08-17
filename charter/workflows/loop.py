"""The generic feedback loop every Charter agent runs.

    entry -> agent -> deliverable   -> [approval] -> Complete
                    \\-> propose     -> AwaitApproval -> approved -> execute_act -> entry
                    \\                                \\-> rejected -> entry (with the reason)
                    \\-> ask_human   -> AwaitInput -> entry (with the answer)

Two operations. `entry` is the only place that stages context and calls the agent —
a fresh start, a resumed answer, a rejection, and a completed action all end up
there, because each is "fold in new information and run again."

Only a deliverable ends a task. An approved act is a step: the tool runs, its result
folds into history, and the loop re-enters, so an agent can take several actions in
one task and always gets to report what it did.

Derived from receiptbot's `question_answerer`, with the receipt-specific parts
replaced by config.
"""

from __future__ import annotations

import logging
from typing import Any

from boundflow import (
    ENTRY_OPERATION,
    AgentDefinition,
    AwaitApproval,
    AwaitInput,
    Budget,
    Complete,
    Next,
    OperationContext,
)
from boundflow.llm import AgentPolicyLimitExceeded

from ..config.agent import AgentConfig
from ..mcp.client import McpError, ToolSet

log = logging.getLogger(__name__)

OP_EXECUTE_ACT = "execute_act"

# Context keys Charter owns. Underscored so they can't collide with an author's
# declared inputs, which share the same dict.
K_ITERATION = "_iteration"
K_COST = "_cost"
K_LLM_CALLS = "_llm_calls"
K_HISTORY = "_history"
K_PROPOSAL = "_proposal"
K_REJECTED = "_rejected"
K_TIMED_OUT = "_timed_out"
K_ACTS = "_acts"
K_FINAL = "_final"
K_DRAFTS = "_drafts"          # submissions a human has rejected
K_QUESTIONS = "_questions"    # questions asked since the last draft
K_TOOL_FAILS = "_tool_fails"  # {tool: failures} across the whole task
K_TOOL_CALLS = "_tool_calls"  # {tool: calls} across the whole task

# Transient LLM errors are near-instant to know about, so a couple of immediate
# in-process attempts is enough before letting it propagate and fail the run for
# real. Same bound question_answerer uses.
MAX_AGENT_ATTEMPTS = 2


async def run_agent_with_retry(ctx: OperationContext, agent: AgentDefinition,
                               budget: Budget | None = None):
    """Run the agent, retrying in place on a *transient* error before letting it
    propagate (at which point there's no automatic retry left).

    AgentPolicyLimitExceeded is deliberately not retried: running out of budget is
    not a transient condition, and a second attempt would spend more to fail the
    same way. question_answerer retries every exception, which is safe there
    because it has no budget to exhaust.
    """
    last: Exception | None = None
    for _ in range(MAX_AGENT_ATTEMPTS):
        try:
            return await ctx.run_agent(agent, budget=budget)
        except AgentPolicyLimitExceeded:
            raise
        except Exception as e:  # noqa: BLE001 — retried, then re-raised below
            last = e
    raise last


class Ended(Exception):
    """A per-run budget is spent. Carries what to tell the operator."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _bullets(lines: list[str]) -> str:
    """A multi-line context payload, indented so its structure survives.

    BoundFlow renders each context entry as `- {label}: {payload}` with the payload
    inlined verbatim, so a bare newline puts the next line flush left where it reads
    as a new entry rather than a continuation.
    """
    return "\n" + "\n".join(f"  - {line}" for line in lines)


def render(template: str, inputs: dict) -> str:
    """`{{ inputs.<name> }}` and nothing else. Validated at apply time, so anything
    unresolved here is a missing invoke context rather than a bad template."""
    out = template
    for name, value in inputs.items():
        for spelling in (f"{{{{ inputs.{name} }}}}", f"{{{{inputs.{name}}}}}"):
            out = out.replace(spelling, str(value))
    return out


def build_output_schema(cfg: AgentConfig) -> dict:
    """The `submit_result` schema — the branch the model fills is what decides what
    happens next.

    `propose` and `ask_human` are injected here, never authored, which is why
    they're reserved names in `deliverable`.
    """
    schema: dict[str, Any] = {}
    for name, spec in cfg.outcome.deliverable.items():
        field: dict[str, Any] = {"type": spec.type}
        if spec.description:
            field["description"] = spec.description
        schema[name] = field

    gated = cfg.gated_tools
    if gated:
        schema["propose"] = {
            "type": "object",
            "description": (
                "Ask a human to run one of these tools on your behalf. You cannot "
                "call them yourself. Set this INSTEAD of the deliverable fields."),
            "properties": {
                "tool": {"type": "string", "enum": gated},
                "args": {"type": "object", "description": "arguments for that tool"},
                "why": {"type": "string", "description": "one line justifying it, shown to the approver"},
            },
        }

    if cfg.outcome.ask_human is not None:
        schema["ask_human"] = {
            "type": "string",
            "description": (
                "A specific question for a human, when you cannot proceed without "
                "an answer. Set this INSTEAD of the deliverable fields."),
        }
    return schema


def build_agent(cfg: AgentConfig, tools: ToolSet, inputs: dict) -> AgentDefinition:
    """Gated tools are absent from `tools` by construction — the model is never
    handed one, which is the safety claim rather than a runtime check."""
    system_prompt = render(cfg.objective, inputs)
    if cfg.outcome.ask_human is not None:
        # Charter owns this wording so it's consistent and tested, rather than each
        # author writing their own weaker version of it.
        system_prompt = f"{system_prompt}\n\n{cfg.outcome.ask_human.guidance}"

    return AgentDefinition(
        name=cfg.name,
        system_prompt=system_prompt,
        model=cfg.model,
        tools=tools.inline_tools(cfg),
        output_schema=build_output_schema(cfg),
        cache=True,
    )


class Loop:
    """One agent version's handlers. Holds no per-task state — everything lives in
    `ctx.context`, so a task can resume on a different worker after a gate."""

    def __init__(self, cfg: AgentConfig, runtime, tools: ToolSet, memory=None) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.tools = tools
        self.memory = memory

    # ── budget, which BoundFlow can't see ───────────────────────────────────

    def _charge(self, ctx: OperationContext, result) -> None:
        c = ctx.context
        c[K_COST] = c.get(K_COST, 0.0) + result.cost_usd
        c[K_LLM_CALLS] = c.get(K_LLM_CALLS, 0) + result.llm_calls_used

    def _operation_timeout(self) -> int:
        """How long the next round may take before the control plane cancels it.

        Derived, not picked. A round is one full agent step — up to `max_llm_calls`
        calls, each bounded by `max_call_seconds` — so a fixed 60s would cancel
        exactly the slow, tool-heavy rounds and leave the fast ones untouched, which
        is the worst possible selection bias for a bug to have.

        Both inputs are already declared by the customer, so this needs no new knob.
        """
        per_run, limits = self.runtime.per_run, self.runtime.limits
        worst_case = (per_run.max_llm_calls or 20) * limits.max_call_seconds
        # A floor so a tiny budget still gets room to dispatch, and a ceiling so a
        # wedged round can't hold its lease for hours.
        return int(min(max(worst_case, 60), 3600))

    def _remaining(self, ctx: OperationContext) -> Budget | None:
        """What's left of the task budget, for this one agent step.

        BoundFlow's runtime policy caps a *single* step and resets every round, so
        without this a six-round task could spend six times its declared budget.
        `Budget` can only tighten the server-side policy, and refuses outright when
        a field reaches zero — which matters because 0 means "unlimited" in
        RuntimePolicy, so passing it through would remove the cap at the exact
        moment it should bite.
        """
        per_run = self.runtime.per_run
        c = ctx.context
        fields: dict[str, Any] = {}
        if per_run.max_cost_usd:
            fields["max_cost_usd"] = per_run.max_cost_usd - c.get(K_COST, 0.0)
        if per_run.max_llm_calls:
            fields["max_llm_calls"] = per_run.max_llm_calls - c.get(K_LLM_CALLS, 0)

        # Per-tool caps reset with every step too, so `max_calls: 5` on a six-round
        # task meant thirty calls until this. Same decrement, per tool.
        used = c.get(K_TOOL_CALLS) or {}
        if per_run.tool_call_limits:
            fields["tool_call_limits"] = {
                l.tool: l.max_calls - used.get(l.tool, 0) for l in per_run.tool_call_limits}

        # Every declared tool gets the same failure allowance; the SDK refuses the
        # tool once it's spent, so a broken integration stops burning the budget
        # mid-round instead of at the next round boundary.
        failed = c.get(K_TOOL_FAILS) or {}
        if per_run.max_tool_failures:
            fields["tool_failure_limits"] = {
                tool: per_run.max_tool_failures - failed.get(tool, 0)
                for tool in self.cfg.all_tools}

        return Budget(**fields) if fields else None

    def _check_limits(self, ctx: OperationContext) -> None:
        """The limits BoundFlow can't see, each naming a distinct reason the task
        isn't converging. Spend is handled by `Budget` at the call itself."""
        per_run = self.runtime.per_run
        c = ctx.context

        if c.get(K_DRAFTS, 0) >= per_run.max_drafts:
            raise Ended(
                f"a human rejected {c[K_DRAFTS]} drafts (max_drafts={per_run.max_drafts}) "
                "— the objective or the agent is wrong for this task")

        for tool, failures in (c.get(K_TOOL_FAILS) or {}).items():
            if failures >= per_run.max_tool_failures:
                raise Ended(
                    f"{tool} failed {failures} times (max_tool_failures="
                    f"{per_run.max_tool_failures}) — the integration looks broken")


    def _fail(self, ctx: OperationContext, reason: str) -> Complete:
        """Record a customer-domain failure and still return a result.

        mark_failed() increments num_failures for the lifecycle rules while the run
        completes normally, so the task reports how far it got instead of dying with
        a stack trace — which matters, because this payload is what an operator
        reads when a rule pauses the agent.
        """
        ctx.mark_failed()
        c = ctx.context
        log.warning("task failed: agent=%s reason=%s", self.cfg.name, reason)
        return Complete(result={
            "failed": True,
            "reason": reason,
            "rounds": c.get(K_ITERATION, 0),
            "cost_usd": round(c.get(K_COST, 0.0), 6),
            "llm_calls": c.get(K_LLM_CALLS, 0),
            # A failed task is not an untouched one: approved acts may already have
            # run. An operator reading this after a lifecycle pause needs to know
            # whether money moved before anything else.
            "acts_performed": c.get(K_ACTS, []),
            "history": c.get(K_HISTORY, []),
        })

    # ── history ─────────────────────────────────────────────────────────────

    def _note(self, ctx: OperationContext, line: str) -> None:
        ctx.context.setdefault(K_HISTORY, []).append(line)

    def _fold_resumption(self, ctx: OperationContext) -> None:
        """Whatever happened while parked, written into history before the agent
        runs again. This is the only reason a rejection teaches the agent anything."""
        reason = ctx.approval_reason
        rejected = ctx.context.pop(K_REJECTED, None)
        if rejected:
            what = rejected.get("what", "your last submission")
            because = f' — they said: "{reason}"' if reason else " (no reason given)"
            self._note(ctx, f"A human REJECTED {what}{because}. Do not repeat it.")
            ctx.context[K_DRAFTS] = ctx.context.get(K_DRAFTS, 0) + 1
            # A fresh allowance: having been told what was wrong, asking again is
            # reasonable in a way that asking twice before any draft is not.
            ctx.context[K_QUESTIONS] = 0

        answer = ctx.input_answer
        if answer is not None:
            question = ctx.context.pop("_question", "your question")
            text = answer.get("text", "") if isinstance(answer, dict) else str(answer)
            self._note(ctx, f'You asked: "{question}" — a human answered: "{text or "(no answer)"}"')

    # ── entry ───────────────────────────────────────────────────────────────

    async def entry(self, ctx: OperationContext):
        c = ctx.context
        c[K_ITERATION] = c.get(K_ITERATION, 0) + 1
        self._fold_resumption(ctx)

        if timed_out := c.pop(K_TIMED_OUT, None):
            return self._fail(ctx, timed_out)

        try:
            self._check_limits(ctx)
        except Ended as e:
            return self._fail(ctx, e.reason)

        # One add_context per block, not per line: each becomes a single labelled
        # bullet in the agent's prompt, so ten history lines would otherwise be ten
        # bullets with the same label. `_bullets` supplies the continuation indent —
        # BoundFlow renders the payload raw, so an unindented second line is
        # indistinguishable from a new top-level entry.
        if self.memory is not None:
            workflow_id = getattr(ctx, "workflow_id", "") or getattr(ctx._op, "workflow_id", "")
            if recalled := await self.memory.recall(self.cfg, workflow_id):
                ctx.add_context("what humans have told you before", _bullets(recalled))
        if history := c.get(K_HISTORY):
            ctx.add_context("what's happened so far", _bullets(history))

        agent = build_agent(self.cfg, self.tools, c)
        try:
            result = await run_agent_with_retry(ctx, agent, self._remaining(ctx))
        except AgentPolicyLimitExceeded as e:
            # The budget was already spent before this round could start. Nothing
            # was called, so there's no partial work to salvage — fail rather than
            # loop into a round that would raise identically.
            return self._fail(ctx, f"task budget exhausted: {e}")

        # Only cost and call counts are accumulated, and only because the per-run
        # budget needs them. Full telemetry belongs to the worker's trace_sink, not
        # to hand-rolled embedding in the context.
        self._charge(ctx, result)

        # Running totals across rounds. These feed the per-tool budgets on the next
        # call, which is what makes `per_run.tool_call_limits` actually per-run —
        # BoundFlow's own caps reset with every step.
        fails = ctx.context.setdefault(K_TOOL_FAILS, {})
        for tool, count in (result.tool_failure_counts or {}).items():
            fails[tool] = fails.get(tool, 0) + count
        calls = ctx.context.setdefault(K_TOOL_CALLS, {})
        for tool, count in (result.calls_per_tool or {}).items():
            calls[tool] = calls.get(tool, 0) + count

        failed = self._fail_fast_hits(result)
        if failed:
            return self._fail(ctx, f"tool {failed} failed and is marked on_failure: fail")

        # Now that this round is paid for: is there anything left to spend? If not,
        # this output is the agent's last word — produced under a forced finalize,
        # not because it was done. We keep it rather than discard it (the agent may
        # have done all the work and simply run short at the end), but every path
        # that surfaces it says so, and nothing loops again.
        c[K_FINAL] = self._spent_out(ctx)

        return await self._dispatch(ctx, result.output or {})

    def _spent_out(self, ctx: OperationContext) -> bool:
        per_run = self.runtime.per_run
        c = ctx.context
        return bool(
            (per_run.max_cost_usd and c.get(K_COST, 0.0) >= per_run.max_cost_usd)
            or (per_run.max_llm_calls and c.get(K_LLM_CALLS, 0) >= per_run.max_llm_calls))

    def _again(self, ctx: OperationContext, note: str):
        """Nudge the agent and go round again — unless there's nothing left to spend,
        in which case another round would only raise on arrival."""
        self._note(ctx, note)
        if ctx.context.get(K_FINAL):
            return self._fail(ctx, f"out of budget with nothing usable submitted ({note})")
        return Next(ENTRY_OPERATION, ctx.context, self._operation_timeout())

    def _truncated_note(self, ctx: OperationContext) -> str:
        """Prepended to any gate whose subject came from a final, forced round. The
        approver decides — but they decide knowing the agent was cut off, which is
        the difference between an informed rejection and a rubber stamp."""
        if not ctx.context.get(K_FINAL):
            return ""
        return ("⚠ produced after this task ran out of budget — the agent was forced "
                "to finalize and may not have finished its reasoning.\n")

    def _fail_fast_hits(self, result) -> str | None:
        """Checked between iterations because the orchestrator swallows tool
        exceptions by design — so the agent may make a few more calls in the current
        iteration before this takes effect."""
        for tool, count in (result.tool_failure_counts or {}).items():
            if count and tool in self.cfg.fail_fast_tools:
                return tool
        return None

    async def _dispatch(self, ctx: OperationContext, output: dict):
        ask = self._asked(ctx, output)
        if ask is not None:
            return ask

        proposal = output.get("propose")
        if proposal and proposal.get("tool"):
            return self._propose(ctx, proposal)

        return self._deliver(ctx, output)

    def _asked(self, ctx: OperationContext, output: dict):
        """Park for a human answer, if the agent asked and still has an allowance.

        Running out of questions isn't a failure — it's "stop asking and show me
        something." So we fall through to the draft branches rather than ending the
        task; the allowance refreshes after each rejection.
        """
        spec = self.cfg.outcome.ask_human
        question = output.get("ask_human")
        if spec is None or not question:
            return None

        asked = ctx.context.get(K_QUESTIONS, 0)
        if asked >= self.runtime.per_run.max_questions:
            self._note(ctx, (
                f"You have asked {asked} questions without submitting anything "
                f"(max_questions={self.runtime.per_run.max_questions}). Submit your "
                "best draft with what you already know."))
            return None

        ctx.context[K_QUESTIONS] = asked + 1

        # Thread what the branch needs through its own context rather than mutating
        # ctx.context first — the parked branch carries its own state.
        answered = Next(ENTRY_OPERATION, {**ctx.context, "_question": question}, self._operation_timeout())
        timed_out = Next(
            ENTRY_OPERATION,
            {**ctx.context, "_question": question,
             **({K_TIMED_OUT: "nobody answered the agent's question"}
                if spec.on_timeout == "fail" else {})},
            self._operation_timeout())
        return AwaitInput(
            on_answer=answered,
            on_timeout=timed_out,
            timeout=spec.timeout_seconds,
            prompt=question,
            metadata=ctx.context)

    def _propose(self, ctx: OperationContext, proposal: dict):
        """The model named a tool it cannot call. Park until a human decides."""
        tool = proposal.get("tool")
        if tool not in self.cfg.gated_tools:
            return self._again(ctx, f"{tool!r} is not a tool you may propose. Choose "
                                    f"one of: {', '.join(self.cfg.gated_tools)}.")

        spec = self.cfg.outcome.approval
        approved = Next(OP_EXECUTE_ACT, {**ctx.context, K_PROPOSAL: proposal}, self._operation_timeout())
        rejected = Next(
            ENTRY_OPERATION,
            {**ctx.context, K_REJECTED: {"what": f"your proposal to call {tool}"}},
            self._operation_timeout())

        return AwaitApproval(
            on_approve=approved,
            on_reject=rejected,
            timeout=spec.timeout_seconds,
            justification=self._justify(ctx, proposal),
            metadata=ctx.context)

    def _justify(self, ctx: OperationContext, proposal: dict) -> str:
        """Composed by Charter, not templated — an author can't ship a gate with the
        amount left out."""
        lines = [f"{self._truncated_note(ctx)}{self.cfg.name}: run {proposal.get('tool')}"]
        for key, value in (proposal.get("args") or {}).items():
            lines.append(f"  {key}: {value}")
        if why := proposal.get("why"):
            lines.append(f"why: {why}")
        if note := (self.cfg.outcome.approval.note if self.cfg.outcome.approval else None):
            lines.append(render(note, ctx.context))
        return "\n".join(lines)

    def _deliver(self, ctx: OperationContext, output: dict):
        deliverable = {k: output.get(k) for k in self.cfg.outcome.deliverable}
        if all(v is None for v in deliverable.values()):
            return self._again(
                ctx, "You submitted nothing usable. Provide the deliverable fields.")

        result = {**deliverable,
                  "rounds": ctx.context.get(K_ITERATION, 0),
                  "cost_usd": round(ctx.context.get(K_COST, 0.0), 6),
                  "acts_performed": ctx.context.get(K_ACTS, [])}
        if ctx.context.get(K_FINAL):
            result["truncated"] = True

        if self.cfg.outcome.deliverable_approval != "always":
            return Complete(result=result)

        spec = self.cfg.outcome.approval
        rejected = Next(
            ENTRY_OPERATION,
            {**ctx.context, K_REJECTED: {"what": "your proposed answer"}},
            self._operation_timeout())
        return AwaitApproval(
            on_approve=Complete(result=result),
            on_reject=rejected,
            timeout=spec.timeout_seconds,
            justification=self._justify_deliverable(ctx, deliverable),
            metadata=ctx.context)

    def _justify_deliverable(self, ctx: OperationContext, deliverable: dict) -> str:
        lines = [f"{self._truncated_note(ctx)}{self.cfg.name} proposes to finish with:"]
        lines += [f"  {k}: {v}" for k, v in deliverable.items()]
        if note := (self.cfg.outcome.approval.note if self.cfg.outcome.approval else None):
            lines.append(render(note, ctx.context))
        return "\n".join(lines)

    # ── execute_act ─────────────────────────────────────────────────────────

    async def execute_act(self, ctx: OperationContext):
        """Run the approved tool. The only place a gated tool is ever called, and it
        runs after the decision rather than inside the agent loop."""
        proposal = ctx.context.pop(K_PROPOSAL, None) or {}
        tool = proposal.get("tool", "")
        reason = ctx.approval_reason

        try:
            output = await self.tools.call_gated(tool, proposal.get("args"))
        except McpError as e:
            self._record_tool_call(ctx, tool, failed=True)
            if tool in self.cfg.fail_fast_tools:
                return self._fail(ctx, f"approved tool {tool} failed: {e}")
            self._note(ctx, f"{tool} was approved but FAILED: {e}")
            return Next(ENTRY_OPERATION, ctx.context, self._operation_timeout())

        self._record_tool_call(ctx, tool, failed=False)
        ctx.context.setdefault(K_ACTS, []).append(
            {"tool": tool, "args": proposal.get("args") or {}})
        approved = f"A human approved {tool} and it ran"
        if reason:
            approved += f' (they said: "{reason}")'
        self._note(ctx, f"{approved}. Result: {output}")
        return Next(ENTRY_OPERATION, ctx.context, self._operation_timeout())

    def _record_tool_call(self, ctx: OperationContext, tool: str, *, failed: bool) -> None:
        """There's no orchestrator here to count for us, so write the snapshot by
        hand — same shape run_agent emits. Without this an approved tool that fails
        every time is invisible to the `tool_failures` rule meant to catch it."""
        import time

        ctx.agent_state_updates[self.cfg.name] = {
            "cost_usd": 0.0,
            "llm_calls": 0,
            "tokens_used": 0,
            "calls_per_tool": {tool: 1},
            "tool_failure_counts": {tool: 1} if failed else {},
            "latency_seconds": 0.0,
            "ran_at": int(time.time() * 1000),
        }
