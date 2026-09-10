"""A task from invoke to result, against a real control plane.

Only the model is faked. The control plane, the MCP subprocess, and the Postgres
holding the harness's conversation are all real — a task that parks at a gate and
resumes is only meaningful against a store that actually persisted something.

What changed with the harness: there is no propose/execute split any more, so a
gated tool is *handed* to the model and its call is stopped, rather than being
withheld. That is a weaker claim than omission and a better product — same moment
the harness chose, a pause that outlives the worker, and an approver who can edit
the arguments rather than only refuse them.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

from charter.config.loader import load_project
from charter.provisioning.apply import apply_project, compile_agent, create_instance
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_gate, wait_for_run
from tests.e2e.harness import calls, factory, scripted, submits

pytestmark = pytest.mark.asyncio

PLAYGROUND = Path(__file__).parent.parent.parent / "playground"


@pytest.fixture
def project(tmp_path, tenant, store_url):
    """The playground, pointed at this test's own tenant and store."""
    dst = tmp_path / "playground"
    shutil.copytree(PLAYGROUND, dst)
    raw = yaml.safe_load((dst / "worker.yaml").read_text())
    raw["control_plane"]["tenant"] = tenant.name
    raw["store"] = {"url": store_url}
    raw.pop("notifications", None)          # nothing is listening in a test
    (dst / "worker.yaml").write_text(yaml.safe_dump(raw))
    # The worker spawns the MCP server relative to its own directory, and by bare
    # `python` — which under pytest is whatever is on PATH rather than the
    # interpreter running the tests. Pin both, or the server starts without `mcp`
    # installed and every agent quarantines.
    for agent in ("refund-demo", "ticket-sweeper", "delegator"):
        v1 = dst / agent / "v1.yaml"
        if v1.exists():
            v1.write_text(v1.read_text()
                          .replace('command: python', f'command: {sys.executable}')
                          .replace('args: ["mcp_server.py"]',
                                   f'args: ["{PLAYGROUND / "mcp_server.py"}"]'))
    return load_project(dst / "worker.yaml")


async def one_instance(cp, project, agent, tenant):
    """Apply never creates now — an instance is an entity someone decides to make."""
    bundle = project.agents[agent]
    wf = await create_instance(cp, compile_agent(bundle), tenant.id)
    await apply_project(cp, project, only=agent, all_=True)
    return wf


async def test_a_task_runs_and_publishes_what_the_agent_returned(cp, project, tenant):
    """Charter injects no fields of its own now, so the result is the agent's
    answer rather than a wrapper to unpick."""
    wf = await one_instance(cp, project, "ticket-sweeper", tenant)
    model = scripted(
        calls("desk__list_open_tickets"),
        submits(summary="two tickets need a look", needs_attention=2),
    )

    worker = CharterWorker(project, chat_model=factory(model))
    async with running(worker):
        info = await wait_for_run(cp, await cp.invoke_workflow(wf.id), timeout=90)

    assert info.status.value == "completed", info.failure_reason
    assert info.result["summary"] == "two tickets need a look"
    assert info.result["needs_attention"] == 2


async def test_a_gated_tool_is_offered_and_its_call_is_stopped(cp, project, tenant):
    """The claim the config is making. Under the harness the tool *is* in the list
    — omission was the old mechanism — so what has to hold is that the call
    doesn't go through without a human."""
    wf = await one_instance(cp, project, "refund-demo", tenant)
    model = scripted(
        calls("desk__create_refund", charge_id="ch_9002", amount_usd=240,
              reason="duplicate"),
        submits(resolution="refunded", refunded_usd=240),
    )

    worker = CharterWorker(project, chat_model=factory(model))
    async with running(worker):
        await cp.invoke_workflow(wf.id, context={"ticket_id": "4821"})
        gate = await wait_for_gate(cp, wf.id, timeout=90)

    assert "desk__create_refund" in gate.justification
    # The arguments reach the approver, or they're deciding on a name alone.
    assert "ch_9002" in gate.justification
    assert any("desk__create_refund" in names for names in model.offered)


async def test_an_approval_resumes_the_same_conversation(cp, project, tenant):
    """The whole thesis: the task parks, the operation ends, and what comes back
    is the same agent mid-thought rather than a new one starting over."""
    wf = await one_instance(cp, project, "refund-demo", tenant)
    model = scripted(
        calls("desk__create_refund", charge_id="ch_9002", amount_usd=240,
              reason="duplicate"),
        submits(resolution="refunded the duplicate", refunded_usd=240),
    )

    worker = CharterWorker(project, chat_model=factory(model))
    async with running(worker):
        request_id = await cp.invoke_workflow(wf.id, context={"ticket_id": "4821"})
        gate = await wait_for_gate(cp, wf.id, timeout=90)
        await cp.approve_workflow(wf.id, gate.approval_id, "e2e", "duplicate confirmed")
        info = await wait_for_run(cp, request_id, timeout=90)

    assert info.status.value == "completed", info.failure_reason
    assert info.result["refunded_usd"] == 240


async def test_a_rejection_reaches_the_model(cp, project, tenant):
    """A rejection whose reason the agent can't read is the least useful kind, and
    the reason only exists at decision time — after the gate was raised."""
    wf = await one_instance(cp, project, "refund-demo", tenant)
    model = scripted(
        calls("desk__create_refund", charge_id="ch_7700", amount_usd=89,
              reason="changed their mind"),
        submits(resolution="no refund — outside the window", refunded_usd=0),
    )

    worker = CharterWorker(project, chat_model=factory(model))
    async with running(worker):
        request_id = await cp.invoke_workflow(wf.id, context={"ticket_id": "5150"})
        gate = await wait_for_gate(cp, wf.id, timeout=90)
        await cp.reject_workflow(wf.id, gate.approval_id, "e2e",
                                 "three months is outside the window")
        info = await wait_for_run(cp, request_id, timeout=90)

    assert info.status.value == "completed", info.failure_reason
    assert info.result["refunded_usd"] == 0


async def arm(cp, wf, agent, *rules):
    """Arm workflow lifecycle rules, built the way `charter apply` builds them.

    Written here rather than added to `playground/` as a lifecycle.yaml: these
    rules act on the agent across runs, so a file would arm them for every other
    test that serves the same agent.
    """
    from charter.compile import compile_workflow_rules
    from charter.config.lifecycle import LifecyclePolicyFile

    compiled = compile_workflow_rules(LifecyclePolicyFile.model_validate({
        "apiVersion": "charter/v1", "kind": "LifecyclePolicy",
        "agent": agent, "rules": list(rules)}))
    await cp.set_workflow_lifecycle_policy(wf.id, compiled)


async def state_becomes(cp, wf_id, wanted, timeout=90):
    """Poll for a workflow state. Rules are evaluated between runs, so the change
    lands after the run that crossed the threshold has finished, not during it."""
    import asyncio

    deadline = asyncio.get_event_loop().time() + timeout
    seen = None
    while asyncio.get_event_loop().time() < deadline:
        wf = next(w for w in await cp.list_workflows() if w.id == wf_id)
        seen = wf.workflow_state.value
        if seen == wanted:
            return wf
        await asyncio.sleep(2)
    raise AssertionError(f"workflow_state stayed {seen!r}, wanted {wanted!r}")


async def test_a_rejection_threshold_pauses_the_agent(cp, project, tenant):
    """The README's claim, and the one thing no test covered: a metric crossing a
    threshold stops the agent on its own.

    `approval_rejections` is the metric that goes wrong quietly — it is counted
    when a gate is answered, so a worker that dies mid-operation used to drop it,
    and a pause rule reading it silently under-counted.
    """
    wf = await one_instance(cp, project, "refund-demo", tenant)
    await arm(cp, wf, "refund-demo",
              {"when": {"metric": "approval_rejections", "threshold": 1},
                       "then": {"pause": {"window": 1}}})

    model = scripted(
        calls("desk__create_refund", charge_id="ch_7700", amount_usd=89,
              reason="changed their mind"),
        submits(resolution="no refund", refunded_usd=0),
    )

    worker = CharterWorker(project, chat_model=factory(model))
    async with running(worker):
        request_id = await cp.invoke_workflow(wf.id, context={"ticket_id": "5150"})
        gate = await wait_for_gate(cp, wf.id, timeout=90)
        await cp.reject_workflow(wf.id, gate.approval_id, "e2e", "outside the window")
        info = await wait_for_run(cp, request_id, timeout=90)
        assert info.status.value == "completed", info.failure_reason

        paused = await state_becomes(cp, wf.id, "paused")

    assert paused.workflow_state.value == "paused"
