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
from boundflow import ENTRY_OPERATION, AwaitApproval, Complete, Next
from boundflow.llm import AgentPolicyLimitExceeded

from charter.config.loader import load_agent
from charter.workflows.loop import (
    K_COST,
    K_DECISION,
    K_GATES,
    K_LLM_CALLS,
    Loop,
    interrupt_on,
    render,
    response_schema,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


class FakeResult:
    def __init__(self, output, cost=0.01, calls=2):
        self.output = output
        self.cost_usd = cost
        self.llm_calls_used = calls


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

    def mark_failed(self):
        self.failed = True

    async def run_governed(self, name, invoke, *, chat_model, tools,
                           output_schema=None, budget=None):
        self.budgets.append(budget)
        nxt = self._results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def loop_for(agent="refund-triage", monkeypatch=None):
    bundle = load_agent(EXAMPLES / agent)
    cfg = bundle.latest
    loop = Loop(cfg, bundle.runtime, tools=[], chat_model=lambda m: object(),
                store_url="postgresql://unused")
    return cfg, loop


@pytest.fixture(autouse=True)
def _stub_harness(monkeypatch):
    """No database, and no deepagents graph — the branch under test is ours."""
    class Wiring:
        thread_id = "task-1"
        wiring: dict = {}
        config: dict = {}

        def first(self, payload):
            return payload

    @asynccontextmanager
    async def fake(ctx, agent_name, store_url, *, resume=None):
        fake.resume = resume
        yield Wiring()

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


def test_response_format_becomes_a_schema_and_is_optional():
    cfg = load_agent(EXAMPLES / "refund-triage").latest
    schema = response_schema(cfg)
    assert set(schema["properties"]) == {"resolution", "refunded_usd"}
    assert schema["additionalProperties"] is False

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
    assert out.justification == "Refund $40 to the customer"
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
