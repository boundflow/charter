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
        "stripe__get_charge": 5,
        "zendesk__search_tickets": 10,
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
    assert rule.tool == "stripe__create_refund"
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


def test_entry_operation_gets_the_same_timeout_as_every_other_round():
    """WorkflowConfig.invoke_timeout_seconds is the entry operation's deadline and
    Next.timeout is every later one's. Left unset it defaults to 60s, so round one
    would be cancelled while rounds two onward had forty minutes."""
    bundle = load_agent(EXAMPLES / "refund-triage")
    compiled = compile_agent(bundle)
    assert (compiled.workflow_config.invoke_timeout_seconds
            == bundle.runtime.operation_timeout_seconds)
    assert compiled.workflow_config.invoke_timeout_seconds == 40 * 60


def test_schedule_becomes_repeat_and_triggerable():
    c = compile_agent(load_agent(EXAMPLES / "ticket-summarizer"), 2)
    assert c.workflow_config.repeat_every_seconds == 900
    assert c.workflow_config.triggerable is True


def test_no_schedule_means_no_repeat():
    assert refund().workflow_config.repeat_every_seconds == 0


# ── the half BoundFlow carries but never reads ───────────────────────────────


def test_charters_own_limits_ride_in_custom():
    """BoundFlow's invariant: every typed field is enforced by its SDK, and nothing
    in `custom` is enforced by anything. Capability caps, file rules and the
    allowlists are enforced by Charter and the harness it wires up, so a typed
    field for them over there would claim an enforcement that doesn't exist — and
    would mean a control plane that knows what a deepagents capability is.
    """
    from charter import policy

    custom = refund().runtime_policy.custom

    assert policy.allowed_capabilities(refund().runtime_policy) == {"read"}
    assert custom["capability_call_limits"], "declared caps should travel"
    assert all({"capability", "max_calls"} == set(l) for l in
               custom["capability_call_limits"])
    # Plain JSON-able data, because the wire treats it as a struct — a model here
    # would only be re-parsed by us on the far side.
    assert isinstance(custom["allowed_capabilities"], list)


def test_writing_and_reading_custom_cannot_drift():
    """Both ends live in charter/policy.py for this reason: a key spelled one way
    when written and another when read is a policy that silently stops applying,
    and nothing anywhere fails."""
    from charter import policy

    compiled = refund().runtime_policy

    assert policy.capability_call_caps(compiled), "written caps must read back"
    assert policy.allowed_tools(compiled), "an allowlist implies the tool list"


def test_the_tool_allowlist_only_appears_alongside_a_capability_one():
    """Empty means "no allowlist", not "nothing permitted". Declared MCP tools are
    always allowed, so the list only ever has to name what the harness brings —
    and setting it without a capability allowlist would forbid everything."""
    from charter import policy

    summary = summarizer().runtime_policy   # declares no allowed_capabilities

    assert policy.allowed_capabilities(summary) == set()
    assert policy.allowed_tools(summary) == set()


def test_a_failure_allowance_is_a_typed_field_because_boundflow_enforces_it():
    """The one that moved the other way: tool_failure_limits raises
    ToolFailureLimitExceeded in the SDK, so it belongs in the typed half."""
    limits = refund().runtime_policy.tool_failure_limits

    assert limits, "declared max_tool_failures should reach the policy"
    assert all(l.max_failures == 3 for l in limits)


def test_a_dying_worker_hands_the_operation_over_rather_than_stopping_the_agent():
    """`resumable` exists for exactly Charter's case — BoundFlow's own note says to
    set it where "a governed harness resumes from its checkpoint", which is what
    durable_harness is.

    Left off, a worker dying mid-round interrupts the workflow and it waits for a
    human to clear the interruption. We hit that for real: a TypeError while
    parking left a task stopped until someone ran `charter resume`. An agent that
    needs an operator because a process died is not a durable agent.
    """
    assert compile_agent(load_agent(EXAMPLES / "refund-triage")).workflow_config.resumable is True
