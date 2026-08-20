"""The one operation, against a fake OperationContext.

There is much less to test than there used to be, which is the point: the loop
Charter drove is gone, and what's left is the part the harness can't do. So this
covers the seam — what the harness is handed, what happens when it asks for a
human, and what stops the task when the budget runs out.

`durable_harness` is stubbed rather than run. It opens a real Postgres store and
checkpointer, and a test that needed a database to check which branch we return
would be testing LangGraph, not us.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from boundflow import ENTRY_OPERATION, AwaitApproval, AwaitInput, Complete, Next
from boundflow.llm import AgentPolicyLimitExceeded

from charter.config.loader import load_agent
from charter.workflows.loop import (
    K_COST,
    K_DECISION,
    K_GATES,
    K_LLM_CALLS,
    ASK_TOOL,
    Loop,
    ask_tool,
    interrupt_on,
    render,
    response_schema,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


class FakeResult:
    def __init__(self, output, cost=0.01, calls=2, tool_failures=None):
        self.output = output
        self.cost_usd = cost
        self.llm_calls_used = calls
        self.tool_failure_counts = tool_failures or {}


class FakeCtx:
    """Just enough OperationContext for the loop."""

    def __init__(self, context=None, results=None, approval_reason=None):
        self.context = context if context is not None else {}
        self._results = list(results or [])
        self.budgets = []
        self.failed = False
        self._approval_reason = approval_reason

    @property
    def approval_reason(self):
        r, self._approval_reason = self._approval_reason, None
        return r

    _answer = None

    @property
    def input_answer(self):
        a, self._answer = self._answer, None
        return a

    def mark_failed(self):
        self.failed = True

    _governor = None

    def agent_governor(self, name):
        if self._governor is None:
            self._governor = type("Gov", (), {"cost_usd": 0.0, "llm_calls": 0})()
        return self._governor

    async def run_governed(self, name, invoke, *, chat_model, tools, model=None,
                           output_schema=None, budget=None):
        self.budgets.append(budget)
        nxt = self._results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def loop_for(agent="refund-triage", monkeypatch=None):
    bundle = load_agent(EXAMPLES / agent)
    cfg = bundle.latest
    empty = type("NoTools", (), {"langchain_tools": lambda self: []})()
    loop = Loop(cfg, bundle.runtime, tools=empty, chat_model=lambda m: object(),
                store_url="postgresql://unused")
    return cfg, loop


@pytest.fixture(autouse=True)
def _stub_harness(monkeypatch):
    """No database, and no deepagents graph — the branch under test is ours."""
    class Wiring:
        thread_id = "task-1"
        wiring: dict = {}
        config: dict = {}
        discarded = False

        def first(self, payload):
            return payload

        async def discard(self):
            # Recorded rather than ignored: whether state is dropped is the whole
            # difference between a finished task and a parked one.
            Wiring.discarded = True

    Wiring.discarded = False

    @asynccontextmanager
    async def fake(ctx, agent_name, store_url, *, resume=None):
        fake.resume = resume
        fake.wiring = Wiring()
        yield fake.wiring

    monkeypatch.setattr("boundflow.harness.durable_harness", fake)
    monkeypatch.setattr("charter.workflows.loop.durable_harness", fake)
    monkeypatch.setattr("charter.workflows.loop.task_context",
                        lambda ctx, extra=None: {**(extra or {})})
    monkeypatch.setattr("deepagents.create_deep_agent",
                        lambda **kw: type("G", (), {"ainvoke": lambda s, *a, **k: None})())
    return fake


def run(coro):
    return asyncio.run(coro)


# ── what the harness is handed ───────────────────────────────────────────────


def test_gated_tools_become_interrupts_not_omissions():
    """The claim changed with the harness: a gated tool is still given to the model,
    and the call is stopped. `edit` is offered so an approver can correct an amount
    rather than reject and hope the next attempt is right."""
    cfg = load_agent(EXAMPLES / "refund-triage").latest
    gates = interrupt_on(cfg)
    assert set(gates) == {"stripe__create_refund"}
    assert gates["stripe__create_refund"]["allowed_decisions"] == [
        "approve", "edit", "reject"]


def test_an_agent_with_nothing_gated_asks_for_no_interrupts():
    cfg = load_agent(EXAMPLES / "ticket-summarizer").latest
    assert interrupt_on(cfg) == {}


def test_response_format_is_a_properties_map_not_a_schema():
    """The SDK wraps this as `{"type": "object", "properties": <this>}`. Handing it
    a full JSON Schema nests `{"type": "object"}` where a field belongs, and the
    provider 400s on the first live call — which is exactly how this was found."""
    cfg = load_agent(EXAMPLES / "refund-triage").latest
    fields = response_schema(cfg)
    assert set(fields) == {"resolution", "refunded_usd"}
    assert fields["refunded_usd"]["type"] == "number"
    assert "type" not in fields or isinstance(fields.get("type"), dict), (
        "a bare `type` key means a whole schema leaked in")

    cfg.response_format = None
    assert response_schema(cfg) is None, "prose is a valid answer"


def test_only_declared_inputs_render():
    assert render("ticket {{ inputs.id }}", {"id": 7}) == "ticket 7"
    assert render("{{ inputs.nope }}", {"id": 7}) == "{{ inputs.nope }}"


# ── the gate, which is the whole reason this operation exists ────────────────


def test_a_pending_action_parks_the_task():
    cfg, loop = loop_for()
    interrupt = {"__interrupt__": [type("I", (), {
        "value": {"action_requests": [
            {"name": "stripe__create_refund", "args": {"amount": 40},
             "description": "Refund $40 to the customer"}]},
        "id": "int-1"})()]}
    ctx = FakeCtx(results=[FakeResult(interrupt)])

    out = run(loop.entry(ctx))
    assert isinstance(out, AwaitApproval)
    assert "stripe__create_refund" in out.justification
    assert "Refund $40 to the customer" in out.justification
    assert out.metadata["tool"] == "stripe__create_refund"
    assert out.timeout == cfg.gate.timeout_seconds
    assert ctx.context[K_GATES] == 1


def test_both_branches_return_to_the_one_operation():
    """There is no second operation any more — a resume is the same handler with a
    decision in context."""
    _, loop = loop_for()
    interrupt = {"__interrupt__": [type("I", (), {
        "value": {"action_requests": [{"name": "t", "args": {}, "description": "d"}]},
        "id": "i"})()]}
    out = run(loop.entry(FakeCtx(results=[FakeResult(interrupt)])))

    for branch in (out.on_approve, out.on_reject):
        assert isinstance(branch, Next)
        assert branch.operation == ENTRY_OPERATION
    assert out.on_approve.context[K_DECISION] == "approve"
    assert out.on_reject.context[K_DECISION] == "reject"


def test_the_approvers_reason_reaches_the_model(_stub_harness):
    """Resolved on resume, not when the gate was raised — at that point no human had
    spoken yet. A rejection the agent can't read the reason for is useless."""
    _, loop = loop_for()
    ctx = FakeCtx(context={K_DECISION: "reject"},
                  approval_reason="too much for a first-time customer",
                  results=[FakeResult({"resolution": "dropped it"})])

    run(loop.entry(ctx))
    resumed = _stub_harness.resume
    assert resumed["decisions"][0]["type"] == "reject"
    assert "first-time customer" in resumed["decisions"][0]["message"]


def test_a_timeout_arrives_as_a_rejection_without_a_reason(_stub_harness):
    """AwaitApproval has approve and reject branches and nothing else, and the
    control plane rejects gates whose timeout passed — so an unanswered gate lands
    here with no decider."""
    _, loop = loop_for()
    ctx = FakeCtx(context={K_DECISION: "reject"}, approval_reason=None,
                  results=[FakeResult({"resolution": "dropped it"})])

    run(loop.entry(ctx))
    assert _stub_harness.resume["decisions"][0]["message"] == "no reason given"


def test_an_approval_resumes_without_a_message(_stub_harness):
    _, loop = loop_for()
    run(loop.entry(FakeCtx(context={K_DECISION: "approve"},
                           results=[FakeResult({"resolution": "refunded"})])))
    assert _stub_harness.resume == {"decisions": [{"type": "approve"}]}


def test_a_fresh_task_resumes_nothing(_stub_harness):
    _, loop = loop_for()
    run(loop.entry(FakeCtx(results=[FakeResult({"resolution": "done"})])))
    assert _stub_harness.resume is None


# ── the budget, which the harness can't see ──────────────────────────────────


def test_spend_accumulates_across_gates():
    """A gate ends the operation and the next one gets a fresh runtime policy, so
    without this a task that stopped four times could spend its budget five times."""
    _, loop = loop_for()
    ctx = FakeCtx(results=[FakeResult({"resolution": "done"}, cost=0.05, calls=7)])
    run(loop.entry(ctx))
    assert ctx.context[K_COST] == pytest.approx(0.05)
    assert ctx.context[K_LLM_CALLS] == 7


def test_the_budget_handed_over_is_what_is_left():
    _, loop = loop_for()
    ctx = FakeCtx(context={K_COST: 0.20, K_LLM_CALLS: 30},
                  results=[FakeResult({"resolution": "done"})])
    run(loop.entry(ctx))

    budget = ctx.budgets[0]
    assert budget.max_cost_usd == pytest.approx(0.10)   # 0.30 declared
    assert budget.max_llm_calls == 10                   # 40 declared


def test_every_declared_tool_gets_a_failure_allowance():
    """No harness equivalent exists, so a broken integration would otherwise burn
    the budget instead of tripping its own breaker."""
    cfg, loop = loop_for()
    ctx = FakeCtx(results=[FakeResult({"resolution": "done"})])
    run(loop.entry(ctx))
    assert set(ctx.budgets[0].tool_failure_limits) == set(cfg.all_tools)


def test_a_spent_budget_fails_the_task_with_a_readable_reason():
    """mark_failed trips num_failures for the lifecycle rules while the run still
    completes, so the payload an operator reads says how far it got."""
    _, loop = loop_for()
    ctx = FakeCtx(context={K_GATES: 2},
                  results=[AgentPolicyLimitExceeded("reached max_cost_usd=0.3")])

    out = run(loop.entry(ctx))
    assert isinstance(out, Complete)
    assert ctx.failed
    assert out.result["failed"] is True
    assert "max_cost_usd" in out.result["reason"]
    assert out.result["gates"] == 2


def test_finishing_returns_the_agents_own_answer():
    """Charter injects no fields of its own now, so what comes back is the result,
    not a wrapper to unpick."""
    _, loop = loop_for()
    out = run(loop.entry(FakeCtx(results=[
        FakeResult({"resolution": "refunded $40", "refunded_usd": 40})])))
    assert isinstance(out, Complete)
    assert out.result == {"resolution": "refunded $40", "refunded_usd": 40}


def test_a_gate_says_what_is_about_to_happen():
    """The harness's own description is "Tool execution requires approval", which
    tells someone paged at 2am nothing — and that is what the first live gate
    actually sent. The tool and its arguments have to lead."""
    _, loop = loop_for()
    interrupt = {"__interrupt__": [type("I", (), {
        "value": {"action_requests": [{
            "name": "desk__create_refund",
            "args": {"charge_id": "ch_9002", "amount_usd": 240},
            "description": "Tool execution requires approval"}]},
        "id": "i"})()]}

    out = run(loop.entry(FakeCtx(results=[FakeResult(interrupt)])))
    assert "desk__create_refund" in out.justification
    assert "ch_9002" in out.justification
    assert "Tool execution requires" not in out.justification


class TestAskHuman:
    """The harness only ever stops at a tool call, so asking has to be a tool.
    Declaring `ask_human:` adds one; omitting the block is the only way to say
    never, so there aren't two ways to say it."""

    def _cfg(self, when=None):
        cfg = load_agent(EXAMPLES / "refund-triage").latest
        if when is not None:
            from charter.config.agent import AskHuman
            cfg.ask_human = AskHuman(when=when)
        return cfg

    def test_no_block_means_no_tool(self):
        cfg = self._cfg()
        assert ask_tool(cfg) is None
        assert ASK_TOOL not in interrupt_on(cfg)

    def test_declaring_it_adds_a_gated_tool(self):
        cfg = self._cfg("eagerly")
        assert ask_tool(cfg).name == ASK_TOOL
        # The only sensible decision on a question is an answer.
        assert interrupt_on(cfg)[ASK_TOOL]["allowed_decisions"] == ["response"]

    def test_posture_reaches_the_prompt_and_is_not_a_number(self):
        """A model's confidence in itself isn't calibrated, so a threshold would
        read as precision that isn't there."""
        _, loop = loop_for()
        loop.cfg = self._cfg("eagerly")
        prompt = loop._prompt(FakeCtx())
        assert loop.cfg.objective.split("\n")[0] in prompt
        assert "When in doubt, ask" in prompt

    def test_a_question_parks_for_an_answer_not_a_verdict(self):
        _, loop = loop_for()
        loop.cfg = self._cfg("balanced")
        interrupt = {"__interrupt__": [type("I", (), {
            "value": {"action_requests": [
                {"name": ASK_TOOL, "args": {"question": "Which charge?"},
                 "description": "Tool execution requires approval"}]},
            "id": "i"})()]}

        out = run(loop.entry(FakeCtx(results=[FakeResult(interrupt)])))
        assert isinstance(out, AwaitInput)
        assert "Which charge?" in out.prompt
        assert out.timeout == loop.cfg.ask_human.timeout_seconds

    def test_an_unanswered_question_tells_the_agent_rather_than_failing(self, _stub_harness):
        """AwaitInput has a real timeout branch, unlike approval — so silence means
        carry on with what you have, not that the task dies."""
        _, loop = loop_for()
        loop.cfg = self._cfg("balanced")
        run(loop.entry(FakeCtx(context={K_DECISION: "unanswered"},
                               results=[FakeResult({"resolution": "guessed"})])))
        msg = _stub_harness.resume["decisions"][0]
        assert msg["type"] == "response"
        assert "Nobody answered" in msg["message"]

    def test_an_answer_reaches_the_model(self, _stub_harness):
        _, loop = loop_for()
        loop.cfg = self._cfg("balanced")
        ctx = FakeCtx(context={K_DECISION: "answer"},
                      results=[FakeResult({"resolution": "done"})])
        ctx._answer = "refund ch_9002"
        run(loop.entry(ctx))
        assert _stub_harness.resume["decisions"][0]["message"] == "refund ch_9002"


def test_a_fail_fast_tool_ends_the_task():
    """`on_failure: fail` went dead when the harness took over the loop — the check
    lived in the loop that was deleted. A declared field that quietly does nothing
    is worse than not having it."""
    cfg, loop = loop_for()
    assert "desk__create_refund" in cfg.fail_fast_tools or cfg.fail_fast_tools

    tool = next(iter(cfg.fail_fast_tools))
    out = run(loop.entry(FakeCtx(results=[
        FakeResult({"resolution": "carried on regardless"},
                   tool_failures={tool: 1})])))

    assert isinstance(out, Complete)
    assert out.result["failed"] is True
    assert tool in out.result["reason"]
    assert "on_failure" in out.result["reason"]


def test_an_ordinary_tool_failure_does_not_end_the_task():
    """Only tools that asked for it. Everything else is the model's problem to work
    around, which is the whole reason `on_failure` is a choice."""
    _, loop = loop_for()
    out = run(loop.entry(FakeCtx(results=[
        FakeResult({"resolution": "worked around it"},
                   tool_failures={"desk__get_ticket": 2})])))

    assert isinstance(out, Complete)
    assert "failed" not in out.result


def test_a_spent_budget_reports_what_it_spent():
    """It was reporting llm_calls: 0 on a task that failed precisely because of
    what it spent — `_charge` only runs when the round returns, and a spent cap
    raises instead."""
    _, loop = loop_for()
    ctx = FakeCtx(results=[AgentPolicyLimitExceeded("reached max_llm_calls=2")])
    gov = ctx.agent_governor("x")
    gov.cost_usd, gov.llm_calls = 0.042, 2

    out = run(loop.entry(ctx))
    assert out.result["llm_calls"] == 2
    assert out.result["cost_usd"] == pytest.approx(0.042)
