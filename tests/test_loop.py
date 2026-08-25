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
    K_SECONDS,
    ASK_TOOL,
    Loop,
    WAIT_TOOL,
    ask_tool,
    interrupt_on,
    wait_tool,
    render,
    response_schema,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


class FakeResult:
    def __init__(self, output, cost=0.01, calls=2, tool_failures=None,
                 tool_calls=None):
        self.output = output
        self.cost_usd = cost
        self.llm_calls_used = calls
        self.tool_failure_counts = tool_failures or {}
        self.calls_per_tool = tool_calls or {}


class FakeCtx:
    """Just enough OperationContext for the loop."""

    # The example agent's objective references these, and a task that doesn't
    # supply them now fails before the harness opens — correctly, but it isn't
    # what any of these tests are about, so every context starts with them filled.
    INPUTS = {"ticket_id": "4821", "max_refund_usd": 500}

    def __init__(self, context=None, results=None, approval_reason=None):
        self.context = {**self.INPUTS, **(context or {})}
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

    monkeypatch.setattr("charter.harness.durable_harness", fake)
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
    assert out.timeout == loop.runtime.authority.approval_timeout_seconds
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
        assert interrupt_on(cfg)[ASK_TOOL]["allowed_decisions"] == ["respond"]

    def test_posture_reaches_the_prompt_and_is_not_a_number(self):
        """A model's confidence in itself isn't calibrated, so a threshold would
        read as precision that isn't there."""
        _, loop = loop_for()
        loop.cfg = self._cfg("eagerly")
        ctx = FakeCtx()
        prompt = loop._prompt(ctx)
        # The rendered line, not the raw one: the objective references an input, so
        # comparing against the template only passes while substitution is broken.
        assert render(loop.cfg.objective, ctx.context).split("\n")[0] in prompt
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
        assert out.timeout == loop.runtime.authority.question_timeout_seconds

    def test_an_unanswered_question_tells_the_agent_rather_than_failing(self, _stub_harness):
        """AwaitInput has a real timeout branch, unlike approval — so silence means
        carry on with what you have, not that the task dies."""
        _, loop = loop_for()
        loop.cfg = self._cfg("balanced")
        run(loop.entry(FakeCtx(context={K_DECISION: "unanswered"},
                               results=[FakeResult({"resolution": "guessed"})])))
        msg = _stub_harness.resume["decisions"][0]
        assert msg["type"] == "respond"
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


@pytest.mark.parametrize("answer,expected", [
    ({"answer": "start with 4821"}, "start with 4821"),
    ({"text": "use the newest"}, "use the newest"),
    ({"unexpected": "shape"}, "{'unexpected': 'shape'}"),
    (None, ""),
])
def test_a_structured_answer_becomes_something_the_model_can_read(answer, expected):
    """`submit_input` takes a dict, and the harness's `respond` carries a message —
    so an answer someone took the trouble to give must not be dropped on the way."""
    from charter.workflows.loop import _answer_text
    assert _answer_text(answer) == expected


class TestWorkingTime:
    """`max_seconds` bounds what the *agent* spends, not wall clock. A task parked
    two days waiting for a human has spent none of it — that is the useful reading,
    since it bounds a runaway agent rather than punishing a slow approver."""

    def _loop(self, seconds):
        bundle = load_agent(EXAMPLES / "refund-triage")
        bundle.runtime.per_run.max_seconds = seconds
        empty = type("NoTools", (), {"langchain_tools": lambda self: []})()
        return Loop(bundle.latest, bundle.runtime, tools=empty,
                    chat_model=lambda m: object(), store_url="postgresql://unused")

    def test_time_already_spent_ends_the_task(self):
        loop = self._loop(60)
        ctx = FakeCtx(context={K_SECONDS: 61.0},
                      results=[FakeResult({"resolution": "done"})])

        out = run(loop.entry(ctx))
        assert out.result["failed"] is True
        assert "max_seconds" in out.result["reason"]
        assert "waiting for a human isn't counted" in out.result["reason"]

    def test_a_task_within_its_allowance_completes(self):
        loop = self._loop(60)
        out = run(loop.entry(FakeCtx(context={K_SECONDS: 1.0},
                                     results=[FakeResult({"resolution": "done"})])))
        assert "failed" not in out.result

    def test_no_ceiling_means_no_ceiling(self):
        loop = self._loop(0)
        out = run(loop.entry(FakeCtx(context={K_SECONDS: 99_999.0},
                                     results=[FakeResult({"resolution": "done"})])))
        assert "failed" not in out.result

    def test_the_round_timeout_never_outlasts_the_task_allowance(self):
        """Otherwise the ceiling would only ever be noticed after it had been
        passed — one round could burn the whole budget before anyone checked."""
        bundle = load_agent(EXAMPLES / "refund-triage")
        bundle.runtime.per_run.max_seconds = 90
        assert bundle.runtime.operation_timeout_seconds <= 90

    def test_gate_timeouts_are_a_different_clock(self):
        """A human's allowance and the agent's are independent — one bounds waiting,
        the other bounds working."""
        loop = self._loop(60)
        # The approver's window is policy — a fact about your team. The agent's
        # working ceiling is policy too, but a different number for a different
        # clock: one bounds waiting, the other bounds working.
        assert loop.runtime.authority.approval_timeout_seconds == 1800
        assert loop.runtime.per_run.max_seconds == 60


class TestGateGranularity:
    """One number for every gated tool is the wrong shape — a refund can wait half
    an hour, a production deploy might need someone in five minutes."""

    def _loop(self, **gate):
        bundle = load_agent(EXAMPLES / "refund-triage")
        for k, v in gate.items():
            setattr(bundle.latest.gate, k, v)
        empty = type("NoTools", (), {"langchain_tools": lambda self: []})()
        return Loop(bundle.latest, bundle.runtime, tools=empty,
                    chat_model=lambda m: object(), store_url="postgresql://unused")

    def _interrupt(self, tool="stripe__create_refund"):
        return {"__interrupt__": [type("I", (), {
            "value": {"action_requests": [
                {"name": tool, "args": {"amount": 40}, "description": "d"}]},
            "id": "i"})()]}

    def test_a_tool_without_an_override_uses_the_agent_default(self):
        loop = self._loop()
        out = run(loop.entry(FakeCtx(results=[FakeResult(self._interrupt())])))
        assert out.timeout == 1800

    def test_a_tool_may_shorten_its_own_gate(self):
        loop = self._loop()
        spec = next(t for s in loop.cfg.mcp for t in s.tools
                    if t.tool == "create_refund")
        spec.approval_timeout_seconds = 300

        out = run(loop.entry(FakeCtx(results=[FakeResult(self._interrupt())])))
        assert out.timeout == 300

    def test_on_reject_continue_lets_the_agent_carry_on(self, _stub_harness):
        loop = self._loop(on_reject="continue")
        run(loop.entry(FakeCtx(context={K_DECISION: "reject"},
                               results=[FakeResult({"resolution": "did what I could"})])))
        assert _stub_harness.resume["decisions"][0]["type"] == "reject"

    def test_on_reject_fail_stops_the_task(self):
        """Otherwise a task nobody approved still reports success, and an operator
        scanning statuses never learns the thing it existed to do didn't happen."""
        loop = self._loop(on_reject="fail")
        ctx = FakeCtx(context={K_DECISION: "reject",
                               "_gated_tool": "stripe__create_refund"},
                      approval_reason="too much",
                      results=[FakeResult({"resolution": "unused"})])

        out = run(loop.entry(ctx))
        assert out.result["failed"] is True
        assert "stripe__create_refund" in out.result["reason"]
        assert "too much" in out.result["reason"]

    def test_an_unanswered_gate_under_fail_says_so(self):
        """A timeout and a reasonless rejection arrive identically, so the message
        has to cover both without claiming to know which."""
        loop = self._loop(on_reject="fail")
        out = run(loop.entry(FakeCtx(context={K_DECISION: "reject"},
                                     approval_reason=None,
                                     results=[FakeResult({"resolution": "unused"})])))
        assert "nobody answered in time" in out.result["reason"]


class TestGatingAnything:
    """Gating was only expressible for declared MCP tools, which left no way to
    approve the thing that matters most — the answer itself."""

    def _cfg(self, tools):
        from charter.config.agent import Gate
        cfg = load_agent(EXAMPLES / "refund-triage").latest
        cfg.gate = Gate(tools=tools)
        return cfg

    def test_the_final_answer_can_be_gated(self):
        """`submit_result` is how an agent finishes, so gating it is how you approve
        what it produced — and the approver sees the structured result, not a
        summary of it."""
        gates = interrupt_on(self._cfg(["submit_result"]))
        assert "submit_result" in gates
        assert gates["submit_result"]["allowed_decisions"] == [
            "approve", "edit", "reject"]

    def test_harness_tools_can_be_gated_too(self):
        gates = interrupt_on(self._cfg(["write_file", "execute"]))
        assert {"write_file", "execute"} <= set(gates)

    def test_declared_mcp_tools_still_gate_themselves(self):
        """Two mechanisms, one for each declaration site — a tool that declares
        `approval: always` shouldn't also need naming here."""
        cfg = self._cfg([])
        assert "stripe__create_refund" in interrupt_on(cfg)

    def test_a_typo_is_refused_rather_than_gating_nothing(self):
        from charter.config.agent import Gate
        with pytest.raises(Exception, match="not a tool the harness provides"):
            Gate(tools=["submit_reslt"])


class TestWaiting:
    """The harness cannot park for time — that needs a scheduler, and it is a
    library. So it names the moment by stopping on a tool call, and BoundFlow does
    the waiting. Same split as approvals and questions."""

    def _loop(self, max="7d"):
        from charter.config.agent import Wait
        bundle = load_agent(EXAMPLES / "refund-triage")
        bundle.latest.wait = Wait(max=max)
        empty = type("NoTools", (), {"langchain_tools": lambda self: []})()
        return Loop(bundle.latest, bundle.runtime, tools=empty,
                    chat_model=lambda m: object(), store_url="postgresql://unused")

    def _asks_to_wait(self, duration="1d"):
        return {"__interrupt__": [type("I", (), {
            "value": {"action_requests": [{
                "name": WAIT_TOOL, "args": {"duration": duration, "why": "checking back"},
                "description": "d"}]},
            "id": "i"})()]}

    def test_no_block_means_no_tool(self):
        _, loop = loop_for()
        assert wait_tool(loop.cfg) is None
        assert WAIT_TOOL not in interrupt_on(loop.cfg)

    def test_waiting_parks_the_task_without_holding_anything_open(self):
        """Not a gate: nobody is being asked for anything, and no approval is
        pending. The job simply isn't dispatched until the time has passed."""
        loop = self._loop()
        out = run(loop.entry(FakeCtx(results=[FakeResult(self._asks_to_wait("1d"))])))

        assert isinstance(out, Next)
        assert out.delay_seconds == 86400
        assert out.operation == ENTRY_OPERATION

    def test_a_longer_wait_is_clamped_not_refused(self):
        """The agent asked to wait; waiting less is closer to what it wanted than
        not waiting at all."""
        loop = self._loop(max="6h")
        out = run(loop.entry(FakeCtx(results=[FakeResult(self._asks_to_wait("30d"))])))
        assert out.delay_seconds == 6 * 3600

    def test_a_mangled_duration_does_not_end_the_task(self):
        loop = self._loop()
        out = run(loop.entry(FakeCtx(results=[FakeResult(self._asks_to_wait("soon"))])))
        assert out.delay_seconds == 3600

    def test_the_agent_is_told_how_long_it_actually_slept(self, _stub_harness):
        """It asked for one thing and may have got another, so it shouldn't have to
        assume."""
        loop = self._loop()
        ctx = FakeCtx(context={K_DECISION: "waited", "_waited_for": 172800},
                      results=[FakeResult({"resolution": "carried on"})])

        run(loop.entry(ctx))
        message = _stub_harness.resume["decisions"][0]["message"]
        assert "2 days has passed" in message

    def test_sleeping_does_not_spend_working_time(self):
        """A task that waits a week has spent none of its max_seconds — that is the
        whole reason the budget measures working time."""
        loop = self._loop()
        ctx = FakeCtx(results=[FakeResult(self._asks_to_wait("7d"))])
        run(loop.entry(ctx))
        assert ctx.context.get(K_SECONDS, 0) < 5


# ── publishing ──────────────────────────────────────────────────────────────


class PublishCtx:
    """Just enough context for the publish path: it reads spend and marks failure.
    Distinct from the module's `FakeCtx`, which drives whole operations."""

    def __init__(self) -> None:
        self.context: dict = {}
        self.failed = False

    def mark_failed(self) -> None:
        self.failed = True


async def _published(loop, output, ctx=None):
    return await loop._publish(ctx or PublishCtx(), output)


@pytest.mark.asyncio
async def test_a_structured_answer_is_published_as_it_stands():
    """The ordinary case: the agent filled the declared schema, so that is the
    result and nothing here touches it."""
    _, loop = loop_for()
    result = await _published(loop, {"resolution": "refunded", "refunded_usd": 240})
    assert result.result == {"resolution": "refunded", "refunded_usd": 240}


@pytest.mark.asyncio
async def test_raw_graph_state_is_never_published():
    """It used to be, and the control plane couldn't encode it — the run failed
    with `Unexpected type`, naming no agent and no cause. Anything unreadable
    enough to be mistaken for a platform bug has to be caught here instead."""
    class Msg:
        content = "I had a look but couldn't decide."

    _, loop = loop_for()
    assert loop.cfg.response_format, "this test needs an agent that declares a shape"
    ctx = PublishCtx()

    result = await _published(loop, {"messages": [Msg()], "files": {}}, ctx)

    assert result.result["failed"] is True
    assert "submit_result" in result.result["reason"]
    assert "couldn't decide" in result.result["reason"], "say what it did say"
    assert ctx.failed


@pytest.mark.asyncio
async def test_an_agent_with_no_declared_shape_publishes_its_prose():
    """Nothing was asked for, so prose is the answer rather than a shortfall."""
    class Msg:
        content = "Nothing needs attention today."

    _, loop = loop_for()
    loop.cfg = loop.cfg.model_copy(update={"response_format": None})

    result = await _published(loop, {"messages": [Msg()], "files": {}})

    assert result.result == {"answer": "Nothing needs attention today."}


# ── inputs the task was never given ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_missing_input_fails_before_the_harness_opens():
    """An agent handed `{{ inputs.topic }}` doesn't notice it's a placeholder.

    It reads the unsubstituted template as part of its instructions and asks
    whoever isn't there to supply the value — burning calls and ending in a
    failure that names the wrong thing ("stopped without calling submit_result").
    Caught here it costs nothing and says what to run.
    """
    _, loop = loop_for()
    ctx = FakeCtx()
    ctx.context = {}          # nothing supplied, unlike the fixture default

    out = await loop.entry(ctx)

    assert isinstance(out, Complete)
    assert out.result["failed"] is True
    assert "ticket_id" in out.result["reason"]
    assert "charter run refund-triage" in out.result["reason"], (
        "the reason should be the command to run, not a description of one")


@pytest.mark.asyncio
async def test_an_input_the_objective_never_mentions_is_not_demanded():
    """Only what the objective actually references. A declared input the prompt
    doesn't use is the author's business, not a reason to refuse the task."""
    _, loop = loop_for()
    ctx = FakeCtx()
    ctx.context = {"ticket_id": "4821"}   # max_refund_usd left out

    missing = loop._unfilled(ctx)

    assert "ticket_id" not in missing
    if "max_refund_usd" in missing:
        assert "{{ inputs.max_refund_usd }}" in loop.cfg.objective, (
            "only demanded because the objective references it")


@pytest.mark.asyncio
async def test_a_declared_default_reaches_the_prompt_however_the_task_arrived():
    """Defaults used to be applied by `charter run` alone, so a task invoked any
    other way — a schedule, another service calling invoke_workflow — rendered the
    objective with a literal `{{ inputs.x }}` in it. The agent then reads the
    placeholder as an instruction, which is the failure this whole guard exists to
    stop."""
    _, loop = loop_for()
    ctx = FakeCtx()
    ctx.context = {"ticket_id": "4821"}      # the optional one left out entirely

    prompt = loop._prompt(ctx)

    assert "{{ inputs." not in prompt, "a placeholder survived into the prompt"
    assert "100" in prompt, "the declared default should be what filled it"
    assert loop._unfilled(ctx) == [], "a default is a value, not a missing input"
