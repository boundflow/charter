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
    AwaitInput,
    Budget,
    Complete,
    Next,
)
from boundflow.harness import durable_harness, task_context
from boundflow.harness_gates import approve, pending_action, reject, respond
from boundflow.llm import AgentPolicyLimitExceeded

import time
from pathlib import Path

from ..config.agent import AgentConfig
from .subagents import subagent_limits

log = logging.getLogger(__name__)

# Context keys Charter owns. Underscored so they can't collide with an author's
# declared inputs, which share the same dict.
K_DECISION = "_decision"
K_COST = "_cost"
K_LLM_CALLS = "_llm_calls"
K_GATES = "_gates"
K_GATED_TOOL = "_gated_tool"
K_SECONDS = "_seconds"   # working time, excluding waits for a human


def render(template: str, inputs: dict) -> str:
    """`{{ inputs.<name> }}` and nothing else. Validated at apply time, so anything
    unresolved here is a missing invoke context rather than a bad template."""
    out = template
    for name, value in inputs.items():
        for spelling in (f"{{{{ inputs.{name} }}}}", f"{{{{inputs.{name}}}}}"):
            out = out.replace(spelling, str(value))
    return out


ASK_TOOL = "ask_human"


def ask_tool(cfg: AgentConfig):
    """A tool whose only job is to stop and ask you something, or None.

    The harness has no built-in way for an agent to ask — `interrupt()` is only
    ever raised from the tool-approval path — so asking has to *be* a tool. Gated
    like any other, except the only decision offered is a written answer.

    It never runs: calling it parks the operation, and the answer comes back as the
    tool's result on the next turn. The body exists because a StructuredTool needs
    one.
    """
    if cfg.ask_human is None:
        return None
    from langchain_core.tools import StructuredTool

    async def _ask(question: str) -> str:
        return "waiting for a human"

    return StructuredTool.from_function(
        coroutine=_ask,
        name=ASK_TOOL,
        description=("Ask the human a question and wait for their answer. Use this "
                     "when proceeding would mean guessing at something they know."),
        args_schema={"type": "object",
                     "properties": {"question": {"type": "string"}},
                     "required": ["question"]},
    )


def interrupt_on(cfg: AgentConfig) -> dict[str, Any]:
    """Which tools stop for a human, in the harness's own vocabulary.

    `edit` is offered wherever the harness allows it: an approver correcting an
    amount is a better outcome than rejecting and hoping the next draft is right,
    and it costs nothing to permit — a decision still has to be made either way.
    """
    gates = {tool: {"allowed_decisions": ["approve", "edit", "reject"]}
             for tool in cfg.gated_tools}
    # Harness tools have no declaration site of their own, so they're named on the
    # gate block. `submit_result` among them: gating how an agent finishes is how
    # you approve what it produced.
    for tool in cfg.gate.tools:
        gates.setdefault(tool, {"allowed_decisions": ["approve", "edit", "reject"]})
    if cfg.ask_human is not None:
        # The only sensible decision on a question is an answer.
        gates[ASK_TOOL] = {"allowed_decisions": ["response"]}
    return gates


def response_schema(cfg: AgentConfig) -> dict | None:
    """The agent's structured answer as a *properties map*, or None for prose.

    A map, not a whole JSON Schema: the SDK builds the tool as
    `{"type": "object", "properties": <this>}`, so handing it a full schema nests
    `{"type": "object"}` where a field belongs and the provider rejects the tool —
    a 400 on the first live call, and nothing before that notices.

    Charter adds no fields of its own any more, so what comes back is the agent's
    result rather than a wrapper to unpick.
    """
    if not cfg.response_format:
        return None
    return {name: {"type": spec.type,
                   **({"description": spec.description} if spec.description else {})}
            for name, spec in cfg.response_format.items()}


def _answer_text(answer) -> str:
    """A human's answer as something the model can read.

    `submit_input` takes a dict, so what arrives is structured — but the harness's
    `respond` decision carries a message. Prefer the obvious keys, and fall back to
    the whole thing rather than dropping an answer someone took the trouble to
    give.
    """
    if answer is None:
        return ""
    if isinstance(answer, dict):
        for key in ("answer", "text", "message", "response"):
            if key in answer:
                return str(answer[key])
        return str(answer)
    return str(answer)


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
                 store_url: str, skills: Path | None = None) -> None:
        self.cfg = cfg
        self.runtime = runtime
        # The ToolSet, not its tools: it hasn't connected yet when the worker builds
        # this, and a quarantined agent reconnects later. Asked per task instead.
        self.tools = tools
        self.chat_model = chat_model  # a factory: (model_name) -> BaseChatModel
        self.store_url = store_url
        # This version's skills directory on disk, or None. Uploaded per task —
        # see `_ship_skills`.
        self.skills = skills

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

    def _charge_from_governor(self, ctx) -> None:
        """Fold in whatever the governor recorded before it stopped us.

        `agent_governor` returns the same instance for the same agent within an
        operation, so this is the running total rather than a fresh one.
        """
        try:
            gov = ctx.agent_governor(self.cfg.name)
        except Exception:  # noqa: BLE001 — reporting the failure matters more
            return
        c = ctx.context
        c[K_COST] = c.get(K_COST, 0.0) + gov.cost_usd
        c[K_LLM_CALLS] = c.get(K_LLM_CALLS, 0) + gov.llm_calls

    # No record here of which gated tools ran. It was tried and removed: the
    # transcript already holds it, keyed by `thread_id` — which *is* the request id
    # the CLI prints — and a count derived from our metering is the lossy copy of
    # something with the actual results in it. `charter transcript <task-id>` is
    # the version worth building, joined against the audit so an approval and the
    # call it authorised appear on one timeline. It needs retention first: today a
    # task's state is discarded the moment it completes.

    def _charge_seconds(self, ctx, started: float) -> None:
        c = ctx.context
        c[K_SECONDS] = c.get(K_SECONDS, 0.0) + (time.monotonic() - started)

    def _out_of_time(self, ctx) -> str | None:
        """Whether the task has used its whole allowance.

        Checked at the round boundary, because the harness owns the loop and there
        is no point inside it where we could stop the agent mid-thought. Within a
        round, `operation_timeout_seconds` is the backstop — it's derived from this
        number when one is set, so a single round can't outlast the whole task's
        budget either.
        """
        cap = self.runtime.per_run.max_seconds
        if not cap:
            return None
        spent = ctx.context.get(K_SECONDS, 0.0)
        if spent < cap:
            return None
        return (f"spent {spent:.0f}s of working time (max_seconds={cap:.0f}) — "
                f"time parked waiting for a human isn't counted, so this is the "
                f"agent itself taking too long")

    def _charge(self, ctx, result) -> None:
        c = ctx.context
        c[K_COST] = c.get(K_COST, 0.0) + result.cost_usd
        c[K_LLM_CALLS] = c.get(K_LLM_CALLS, 0) + result.llm_calls_used

    async def _fail(self, ctx, reason: str) -> Complete:
        """Record a customer-domain failure and still return a result.

        `mark_failed()` increments num_failures for the lifecycle rules while the run
        completes normally, so the task reports how far it got instead of dying with
        a stack trace — which matters, because this payload is what an operator reads
        when a rule pauses the agent.
        """
        ctx.mark_failed()
        # From the governor rather than context, because context is only written
        # when the round completes — and a task that fails on a spent budget never
        # gets there. It was reporting `llm_calls: 0` on a run that failed
        # precisely because of what it spent.
        self._charge_from_governor(ctx)
        log.warning("task failed: agent=%s reason=%s", self.cfg.name, reason)
        try:
            async with durable_harness(ctx, self.cfg.name, self.store_url) as h:
                await h.discard()
        except Exception:  # noqa: BLE001 — reporting the failure matters more
            log.warning("could not discard state for the failed task", exc_info=True)
        return Complete(result={
            "failed": True,
            "reason": reason,
            "cost_usd": round(ctx.context.get(K_COST, 0.0), 6),
            "llm_calls": ctx.context.get(K_LLM_CALLS, 0),
            "gates": ctx.context.get(K_GATES, 0),
            "seconds": round(ctx.context.get(K_SECONDS, 0.0), 1),
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
            if self.cfg.gate.on_reject == "fail":
                # The gated action was the point, so finishing without it would be
                # a task that reports success having done nothing. Covers a timeout
                # too — the control plane resolves an unanswered gate as a
                # rejection, and nothing here can tell the two apart.
                refused = ctx.context.pop(K_GATED_TOOL, "the gated action")
                because = ctx.approval_reason or "nobody answered in time"
                return await self._fail(
                    ctx, f"{refused} was not approved ({because}) and this agent is "
                         f"declared on_reject: fail")
            decision = reject(ctx.approval_reason or "no reason given")
        elif verdict == "answer":
            decision = respond(_answer_text(ctx.input_answer))
        elif verdict == "unanswered":
            # Told, not failed: a question nobody answered shouldn't end the task
            # when the agent may well be able to proceed on what it has.
            decision = respond("Nobody answered in time — proceed with what you "
                               "have and say what you assumed.")

        # Working time only. The clock starts when the round starts, so whatever a
        # human took to answer the last gate isn't charged to the agent.
        started = time.monotonic()
        action = None
        try:
            async with durable_harness(ctx, self.cfg.name, self.store_url,
                                       resume=decision) as h:
                skills = await self._ship_skills(h)
                prompt = self._prompt(ctx)
                result = await ctx.run_governed(
                    self.cfg.name,
                    lambda model, tools: create_deep_agent(
                        model=model,
                        tools=tools,
                        system_prompt=prompt,
                        # Ours to declare, theirs to enforce, durable because of us.
                        interrupt_on=interrupt_on(self.cfg),
                        skills=skills or None,
                        **self._wiring(h),
                    ).ainvoke(h.first({"messages": [{"role": "user", "content": prompt}]}),
                              h.config),
                    chat_model=self.chat_model(self.cfg.model),
                    # Passed explicitly rather than sniffed off the chat model.
                    # The versioned config is where the model id actually lives, and
                    # `_derive_model_name` reads provider-specific attributes — so
                    # an integration that doesn't expose one leaves BoundFlow unable
                    # to price the call, which silently disables every cost cap.
                    model=self.cfg.model,
                    tools=self._tools(),
                    output_schema=response_schema(self.cfg),
                    budget=self._remaining(ctx),
                )
                # Decided in here, because only the open harness can drop its own
                # state, and only it knows the key. A task that is parking keeps
                # everything — that state is the entire reason it can resume.
                action = pending_action(result)
                if action is None:
                    await h.discard()
        except AgentPolicyLimitExceeded as spent:
            self._charge_seconds(ctx, started)
            return await self._fail(ctx, str(spent))
        except Ended as ended:
            self._charge_seconds(ctx, started)
            return await self._fail(ctx, ended.reason)

        self._charge(ctx, result)
        self._charge_seconds(ctx, started)

        if (over := self._out_of_time(ctx)) is not None:
            return await self._fail(ctx, over)

        # `on_failure: fail` was declared and unenforced after the harness rewrite
        # — the check went with the loop it lived in. A tool whose failure should
        # end the task has to end it, or the field is a lie in the config.
        if broken := self._fail_fast(result):
            return await self._fail(
                ctx, f"{broken} failed and is declared on_failure: fail — the task "
                     f"stopped rather than working around it")

        if action is not None:
            return self._gate(ctx, action)

        return Complete(result=result.output)

    SKILLS_ROOT = "/skills"

    async def _ship_skills(self, h) -> list[str]:
        """Upload this version's skills into the store, and say where they went.

        The harness reads skills from its *backend*, not from disk — so a directory
        sitting next to the yaml reaches the agent only if we put it there. Loading
        them and never shipping them was the gap: declared, parsed, and invisible.

        Re-uploaded each task rather than cached, because a finished task drops its
        namespace. They are small, and the alternative is a second lifetime to
        reason about.
        """
        if self.skills is None or not self.skills.is_dir():
            return []
        backend = h.wiring.get("backend")
        if backend is None:
            return []

        for path in sorted(self.skills.rglob("*")):
            if path.is_file():
                target = f"{self.SKILLS_ROOT}/{path.relative_to(self.skills)}"
                await backend.awrite(target, path.read_text())
        return [f"{self.SKILLS_ROOT}/"]

    def _wiring(self, h) -> dict:
        """The harness's wiring, plus our subagent bounds.

        Appended rather than replacing: `h.wiring` carries the policy BoundFlow
        already translated — permissions, capability caps — and this is the one
        limit that has no harness equivalent, because to the harness `task` is
        just a tool.
        """
        wiring = dict(h.wiring)
        if extra := subagent_limits(self.runtime.per_run):
            wiring["middleware"] = list(wiring.get("middleware") or []) + extra
        return wiring

    def _tools(self) -> list:
        """Declared MCP tools, plus the ask tool when this agent may ask."""
        tools = self.tools.langchain_tools()
        if (ask := ask_tool(self.cfg)) is not None:
            tools.append(ask)
        return tools

    def _prompt(self, ctx) -> str:
        """The objective, plus how readily this agent should interrupt you.

        Charter owns the wording so it's consistent and testable rather than each
        author writing a weaker version. Rendered with the objective so a document
        can't become a second, quieter set of rules.
        """
        parts = [self.cfg.objective]
        if self.cfg.ask_human is not None:
            parts.append(self.cfg.ask_human.guidance)
        return render("\n\n".join(parts), ctx.context)

    def _fail_fast(self, result) -> str | None:
        """The first declared fail-fast tool that failed this round, if any.

        Checked at the round boundary rather than at the call: the harness owns the
        loop now, so there is no point inside it where we could stop the agent
        mid-thought — and letting it finish the round means its own account of what
        happened reaches the operator.
        """
        failed = getattr(result, "tool_failure_counts", None) or {}
        for tool in self.cfg.fail_fast_tools:
            if failed.get(tool):
                return tool
        return None

    def _gate(self, ctx, action: dict):
        """Park for a human on the action the harness stopped at.

        Only the first pending action is presented even when a turn proposes
        several: a workflow holds one gate at a time, so the rest are decided on
        subsequent rounds rather than lost.
        """
        c = ctx.context
        c[K_GATES] = c.get(K_GATES, 0) + 1
        gate = self.cfg.gate

        tool = action.get("name", "")

        def resume(verdict: str):
            return Next(ENTRY_OPERATION,
                        context=task_context(ctx, {**c, K_DECISION: verdict,
                                                   K_GATED_TOOL: tool}),
                        timeout=self._operation_timeout())

        if action.get("name") == ASK_TOOL:
            return self._question(ctx, action, resume)

        log.info("gate: agent=%s tool=%s", self.cfg.name, action.get("name", "?"))
        return AwaitApproval(
            on_approve=resume("approve"),
            on_reject=resume("reject"),
            timeout=self._gate_timeout(tool),
            justification=self._justify(action),
            metadata={"tool": action.get("name", ""), "args": action.get("args", {})},
        )

    def _gate_timeout(self, tool: str) -> int:
        """How long this tool's approver has.

        Per tool where it's declared, falling back to the agent's default: a refund
        can wait half an hour, a production deploy might need someone in five
        minutes, and one number for both is the wrong shape.
        """
        for server in self.cfg.mcp:
            for spec in server.tools:
                if server.qualified(spec.tool) == tool and spec.approval_timeout_seconds:
                    return spec.approval_timeout_seconds
        return self.cfg.gate.timeout_seconds

    def _question(self, ctx, action: dict, resume):
        """The agent asked something, so park for an answer rather than a verdict.

        `AwaitInput` rather than `AwaitApproval` because there's nothing to approve
        — and unlike approval, it has a real timeout branch, so nobody answering
        means the agent is told and carries on instead of the task dying.
        """
        asked = (action.get("args") or {}).get("question", "")
        log.info("question: agent=%s", self.cfg.name)
        ask = self.cfg.ask_human
        return AwaitInput(
            on_answer=resume("answer"),
            on_timeout=resume("unanswered"),
            timeout=ask.timeout_seconds,
            prompt=f"{self.cfg.name} asks: {asked}" if asked else f"{self.cfg.name} has a question",
            metadata={"question": asked},
        )

    def _justify(self, action: dict) -> str:
        """What the approver reads, and what a notification carries.

        The harness supplies a description, but it is generic — "Tool execution
        requires approval" tells someone paged at 2am nothing, which is what the
        first live gate actually sent. So the tool and its arguments lead, and the
        harness's line is appended only when it says something we didn't.
        """
        name = action.get("name", "a tool")
        args = action.get("args") or {}
        detail = ", ".join(f"{k}={v!r}" for k, v in args.items())
        line = f"{self.cfg.name} wants to call {name}"
        if detail:
            line += f" with {detail}"

        described = (action.get("description") or "").strip()
        if described and not described.lower().startswith("tool execution requires"):
            return f"{line}\n\n{described}"
        return line


def operations(loop: Loop) -> dict:
    """Every operation this workflow has. One, now."""
    return {ENTRY_OPERATION: loop.entry}
