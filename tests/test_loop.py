"""The feedback loop, against a fake OperationContext.

Covers the branches that decide what an agent is allowed to do: what the model is
handed, what it can only propose, what happens when the budget runs out mid-task,
and what a human is shown at a gate.
"""

import asyncio
from pathlib import Path

import pytest
from boundflow import AwaitApproval, AwaitInput, Complete, Next
from boundflow.llm import AgentPolicyLimitExceeded

from charter.config.loader import load_agent
from charter.mcp.client import McpError
from boundflow import ENTRY_OPERATION

from charter.workflows.loop import K_ACTS, K_COST, K_FINAL, K_ITERATION, Loop

EXAMPLES = Path(__file__).parent.parent / "examples"


class FakeResult:
    def __init__(self, output, cost=0.01, calls=2, failures=None, tool_calls=None):
        self.output = output
        self.cost_usd = cost
        self.llm_calls_used = calls
        self.tool_failure_counts = failures or {}
        self.calls_per_tool = tool_calls or {}


class FakeCtx:
    """Just enough OperationContext for the loop."""

    def __init__(self, context=None, results=None, approval_reason=None, answer=None):
        self.context = context if context is not None else {}
        self._results = list(results or [])
        self.budgets = []
        self.staged = []
        self.failed = False
        self.agent_state_updates = {}
        self._approval_reason = approval_reason
        self._answer = answer

    @property
    def approval_reason(self):
        r, self._approval_reason = self._approval_reason, None
        return r

    @property
    def input_answer(self):
        a, self._answer = self._answer, None
        return a

    def add_context(self, label, payload):
        self.staged.append((label, payload))

    def mark_failed(self):
        self.failed = True

    async def run_agent(self, agent, *, budget=None):
        self.budgets.append(budget)
        out = self._results.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeTools:
    def __init__(self, result="done", error=None):
        self.result, self.error, self.calls = result, error, []

    def inline_tools(self, cfg):
        return []

    async def call_gated(self, tool, args):
        self.calls.append((tool, args))
        if self.error:
            raise self.error
        return self.result


def loop(tools=None, **runtime_overrides):
    bundle = load_agent(EXAMPLES / "refund-triage")
    for k, v in runtime_overrides.items():
        setattr(bundle.runtime.per_run, k, v)
    return Loop(bundle.latest, bundle.runtime, tools or FakeTools())


def run(coro):
    return asyncio.run(coro)


DELIVERABLE = {"resolution": "refunded in full", "refunded_usd": 240}
PROPOSAL = {"propose": {"tool": "stripe.create_refund",
                        "args": {"amount": 240}, "why": "duplicate charge"}}


class TestBudget:
    def test_remaining_is_passed_to_run_agent(self):
        """BoundFlow's policy caps one step and resets each round, so without this a
        six-round task could spend six times its budget."""
        lp = loop()
        ctx = FakeCtx({K_COST: 0.10, "_llm_calls": 12}, [FakeResult(DELIVERABLE)])
        run(lp.entry(ctx))
        budget = ctx.budgets[0]
        assert budget.max_cost_usd == pytest.approx(0.30 - 0.10)
        assert budget.max_llm_calls == 40 - 12

    def test_exhausted_budget_fails_without_retrying(self):
        """Running out of budget is not transient — a second attempt would spend
        more to fail identically."""
        lp = loop()
        ctx = FakeCtx({}, [AgentPolicyLimitExceeded("no max_cost_usd budget left")])
        result = run(lp.entry(ctx))
        assert isinstance(result, Complete)
        assert ctx.failed and "budget exhausted" in result.result["reason"]

    def test_transient_error_is_retried(self):
        lp = loop()
        ctx = FakeCtx({}, [ConnectionError("blip"), FakeResult(DELIVERABLE)])
        run(lp.entry(ctx))
        assert len(ctx.budgets) == 2  # retried in place, not failed

    def test_out_of_drafts_fails_with_a_diagnosis(self):
        """Not "it went round a lot" — the reason names what an operator should
        look at."""
        lp = loop(max_drafts=2)
        ctx = FakeCtx({"_drafts": 2}, [FakeResult(DELIVERABLE)])
        result = run(lp.entry(ctx))
        assert ctx.failed
        assert "rejected 2 drafts" in result.result["reason"]
        assert "objective or the agent is wrong" in result.result["reason"]

    def test_repeated_tool_failure_fails_naming_the_tool(self):
        lp = loop(max_tool_failures=3)
        ctx = FakeCtx({"_tool_fails": {"stripe.create_refund": 3}}, [FakeResult(DELIVERABLE)])
        result = run(lp.entry(ctx))
        assert ctx.failed
        assert "stripe.create_refund failed 3 times" in result.result["reason"]
        assert "integration looks broken" in result.result["reason"]


    def test_per_tool_call_limits_are_per_task_not_per_round(self):
        """BoundFlow's own per-tool caps reset every step, so `max_calls: 5` on a
        six-round task used to mean thirty calls."""
        lp = loop()
        ctx = FakeCtx({"_tool_calls": {"stripe.get_charge": 4}}, [FakeResult(DELIVERABLE)])
        run(lp.entry(ctx))
        assert ctx.budgets[0].tool_call_limits["stripe.get_charge"] == 5 - 4

    def test_per_tool_failure_allowance_is_passed_down(self):
        lp = loop()
        ctx = FakeCtx({"_tool_fails": {"stripe.create_refund": 2}}, [FakeResult(DELIVERABLE)])
        run(lp.entry(ctx))
        limits = ctx.budgets[0].tool_failure_limits
        assert limits["stripe.create_refund"] == 3 - 2
        assert limits["zendesk.get_ticket"] == 3  # untouched tools keep the full allowance

    def test_tool_failure_limit_is_not_retried(self):
        """ToolFailureLimitExceeded subclasses AgentPolicyLimitExceeded — retrying a
        broken tool would double its failures before propagating."""
        from boundflow.llm import ToolFailureLimitExceeded
        lp = loop()
        ctx = FakeCtx({}, [ToolFailureLimitExceeded("stripe.create_refund", 3, 3)])
        result = run(lp.entry(ctx))
        assert len(ctx.budgets) == 1  # not retried
        assert ctx.failed and "stripe.create_refund" in result.result["reason"]

    def test_spending_the_last_of_the_budget_marks_the_round_final(self):
        lp = loop()
        ctx = FakeCtx({K_COST: 0.29}, [FakeResult(DELIVERABLE, cost=0.05)])
        run(lp.entry(ctx))
        assert ctx.context[K_FINAL] is True


class TestOutcomes:
    def test_deliverable_completes(self):
        lp = loop()
        lp.cfg.outcome.deliverable_approval = "never"
        ctx = FakeCtx({}, [FakeResult(DELIVERABLE)])
        result = run(lp.entry(ctx))
        assert isinstance(result, Complete)
        assert result.result["resolution"] == "refunded in full"

    def test_deliverable_approval_gates_instead_of_completing(self):
        lp = loop()  # example config sets deliverable_approval: always
        ctx = FakeCtx({}, [FakeResult(DELIVERABLE)])
        result = run(lp.entry(ctx))
        assert isinstance(result, AwaitApproval)
        assert "refunded in full" in result.justification

    def test_proposal_parks_for_approval(self):
        lp = loop()
        ctx = FakeCtx({}, [FakeResult(PROPOSAL)])
        result = run(lp.entry(ctx))
        assert isinstance(result, AwaitApproval)
        # The approver is shown the tool, its arguments, and the agent's reasoning —
        # Charter renders all three, so a gate can't be authored with the amount left out.
        assert "stripe.create_refund" in result.justification
        assert "240" in result.justification
        assert "duplicate charge" in result.justification

    def test_proposing_an_ungated_tool_is_refused(self):
        """execute_act must not become a way to run anything the model names."""
        lp = loop()
        ctx = FakeCtx({}, [FakeResult({"propose": {"tool": "stripe.get_charge", "args": {}}})])
        result = run(lp.entry(ctx))
        assert isinstance(result, Next)
        assert "not a tool you may propose" in ctx.context["_history"][-1]

    def test_ask_human_parks_for_input(self):
        lp = loop()
        ctx = FakeCtx({}, [FakeResult({"ask_human": "Is this the right charge?"})])
        result = run(lp.entry(ctx))
        assert isinstance(result, AwaitInput)
        assert result.prompt == "Is this the right charge?"

    def test_questions_are_allowed_up_to_the_limit(self):
        lp = loop(max_questions=2)
        ctx = FakeCtx({"_questions": 1}, [FakeResult({"ask_human": "Which charge?"})])
        assert isinstance(run(lp.entry(ctx)), AwaitInput)
        assert ctx.context["_questions"] == 2

    def test_running_out_of_questions_forces_a_draft_rather_than_failing(self):
        """Out of questions means "stop asking and show me something", not "give up"."""
        lp = loop(max_questions=2)
        ctx = FakeCtx({"_questions": 2}, [FakeResult({"ask_human": "Which charge?"})])
        result = run(lp.entry(ctx))
        assert not isinstance(result, AwaitInput)
        assert any("Submit your best draft" in line for line in ctx.context["_history"])

    def test_a_rejection_consumes_a_draft_and_refreshes_questions(self):
        """Having been told what was wrong, asking again is reasonable in a way
        that asking twice before any draft is not."""
        lp = loop()
        ctx = FakeCtx({"_drafts": 0, "_questions": 2, "_rejected": {"what": "your answer"}},
                      [FakeResult(DELIVERABLE)], approval_reason="wrong charge")
        run(lp.entry(ctx))
        assert ctx.context["_drafts"] == 1
        assert ctx.context["_questions"] == 0


class TestTruncation:
    def test_final_round_labels_the_gate(self):
        """A human approving a refund must know the agent was cut off — that's the
        difference between an informed rejection and a rubber stamp."""
        lp = loop()
        ctx = FakeCtx({K_COST: 0.29}, [FakeResult(PROPOSAL, cost=0.05)])
        result = run(lp.entry(ctx))
        assert isinstance(result, AwaitApproval)
        assert "ran out of budget" in result.justification

    def test_final_round_marks_the_result(self):
        lp = loop()
        lp.cfg.outcome.deliverable_approval = "never"
        ctx = FakeCtx({K_COST: 0.29}, [FakeResult(DELIVERABLE, cost=0.05)])
        assert run(lp.entry(ctx)).result["truncated"] is True

    def test_final_round_with_nothing_usable_fails_instead_of_looping(self):
        lp = loop()
        ctx = FakeCtx({K_COST: 0.29}, [FakeResult({}, cost=0.05)])
        result = run(lp.entry(ctx))
        assert isinstance(result, Complete) and ctx.failed


class TestResumption:
    def test_rejection_with_a_reason_teaches_the_agent(self):
        """The only reason a rejection is worth anything to the next round."""
        lp = loop()
        ctx = FakeCtx({"_rejected": {"what": "your proposal to call stripe.create_refund"}},
                      [FakeResult(DELIVERABLE)], approval_reason="wrong charge")
        run(lp.entry(ctx))
        note = ctx.context["_history"][0]
        assert "REJECTED" in note and "wrong charge" in note

    def test_rejection_without_a_reason_still_records(self):
        lp = loop()
        ctx = FakeCtx({"_rejected": {"what": "your answer"}}, [FakeResult(DELIVERABLE)])
        run(lp.entry(ctx))
        assert "no reason given" in ctx.context["_history"][0]

    def test_answer_folds_into_history(self):
        lp = loop()
        ctx = FakeCtx({"_question": "Which charge?"}, [FakeResult(DELIVERABLE)],
                      answer={"text": "the March one"})
        run(lp.entry(ctx))
        assert "the March one" in ctx.context["_history"][0]


class TestExecuteAct:
    def test_approved_tool_runs_and_loops_back(self):
        tools = FakeTools(result="refund re_123 created")
        lp = loop(tools=tools)
        ctx = FakeCtx({"_proposal": {"tool": "stripe.create_refund", "args": {"amount": 240}}})
        result = run(lp.execute_act(ctx))
        assert tools.calls == [("stripe.create_refund", {"amount": 240})]
        # Not terminal: the agent sees the result and can act again or finish.
        assert isinstance(result, Next) and result.operation == ENTRY_OPERATION
        assert "refund re_123 created" in ctx.context["_history"][-1]

    def test_performed_acts_are_recorded(self):
        """A failed task is not an untouched one — an operator needs to know whether
        money moved before anything else."""
        lp = loop()
        ctx = FakeCtx({"_proposal": {"tool": "stripe.create_refund", "args": {"amount": 240}}})
        run(lp.execute_act(ctx))
        assert ctx.context[K_ACTS] == [{"tool": "stripe.create_refund", "args": {"amount": 240}}]

    def test_failure_of_a_fail_fast_tool_fails_the_task(self):
        lp = loop(tools=FakeTools(error=McpError("card declined")))
        ctx = FakeCtx({"_proposal": {"tool": "stripe.create_refund", "args": {}}})
        result = run(lp.execute_act(ctx))
        assert isinstance(result, Complete) and ctx.failed
        assert "card declined" in result.result["reason"]

    def test_tool_failure_is_recorded_for_lifecycle_rules(self):
        """No orchestrator here to count for us — without this hand-written
        snapshot, an approved tool failing every time is invisible to the
        tool_failures rule meant to catch it."""
        lp = loop(tools=FakeTools(error=McpError("boom")))
        ctx = FakeCtx({"_proposal": {"tool": "stripe.create_refund", "args": {}}})
        run(lp.execute_act(ctx))
        snap = ctx.agent_state_updates["refund-triage"]
        assert snap["tool_failure_counts"] == {"stripe.create_refund": 1}
