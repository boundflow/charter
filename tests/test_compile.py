from pathlib import Path

from boundflow import Cooldown, InvokeMode, Pause, SetVersion, WorkflowMetric

from charter.compile import compile_agent
from charter.config.loader import load_agent

EXAMPLES = Path(__file__).parent.parent / "examples"


def refund():
    return compile_agent(load_agent(EXAMPLES / "refund-triage"))


def summarizer(version=None):
    return compile_agent(load_agent(EXAMPLES / "ticket-summarizer"), version)


def test_workflow_config():
    c = refund()
    assert c.name == "refund-triage"
    assert c.workflow_config.version == 1
    # Declares inputs -> discrete tasks -> queue, or a ticket gets discarded.
    assert c.workflow_config.invoke_mode == InvokeMode.QUEUE


def test_invoke_mode_coalesces_without_inputs():
    assert summarizer().workflow_config.invoke_mode == InvokeMode.COALESCE


def test_defaults_to_latest_version():
    assert summarizer().version == 2
    assert summarizer(1).version == 1


def test_runtime_policy():
    p = refund().runtime_policy
    # per_run budgets become BoundFlow's per-invocation caps verbatim — deliberate
    # double enforcement, with Charter accumulating the same numbers across
    # iterations.
    assert p.max_cost_usd == 0.30
    assert p.max_llm_calls == 40
    # limits are the per-call valves, not budgets
    assert p.max_tokens_per_call == 1024
    assert p.max_call_seconds == 60
    assert {l.tool: l.max_calls for l in p.tool_call_limits} == {
        "stripe.get_charge": 5,
        "zendesk.search_tickets": 10,
    }


def test_model_is_not_set_on_runtime_policy():
    """The model lives in the versioned config and rides on AgentDefinition.
    Setting it here would make the runtime policy a second source of truth, and a
    policy edit could then change behavior without a version bump."""
    assert refund().runtime_policy.model is None


def test_convergence_limits_have_no_boundflow_equivalent():
    """max_drafts / max_questions / max_tool_failures are Charter's — the control
    plane has no view of the loop."""
    policy = refund().runtime_policy
    for field in ("max_drafts", "max_questions", "max_tool_failures"):
        assert not hasattr(policy, field)


def test_workflow_rules():
    rules = {r.metric: r for r in refund().workflow_rules}

    failures = rules[WorkflowMetric.NUM_FAILURES]
    assert failures.threshold == 2
    assert isinstance(failures.action, Pause)
    assert failures.action.window == 5

    cost = rules[WorkflowMetric.COST]
    assert isinstance(cost.action, Cooldown)
    assert cost.action.seconds == 300

    rejections = rules[WorkflowMetric.APPROVAL_REJECTIONS]
    assert isinstance(rejections.action, SetVersion)
    assert rejections.action.target == 1


def test_tool_failures_renames_to_boundflows_misnomer():
    """Charter says `tool_failures` because the engine compares a summed count, not
    a ratio. BoundFlow's metric is named TOOL_FAILURE_RATE."""
    rule = next(r for r in refund().workflow_rules
                if r.metric == WorkflowMetric.TOOL_FAILURE_RATE)
    assert rule.tool == "stripe.create_refund"
    assert rule.threshold == 3


def test_tool_is_only_set_for_tool_rules():
    for rule in refund().workflow_rules:
        if rule.metric != WorkflowMetric.TOOL_FAILURE_RATE:
            assert rule.tool is None


def test_agent_lifecycle_policy_is_never_produced():
    """Charter exposes workflow lifecycle only — so the effective runtime policy
    always equals what runtime.yaml says."""
    compiled = refund()
    assert not hasattr(compiled, "agent_rules")
    assert compiled.agent_name == "refund-triage"
