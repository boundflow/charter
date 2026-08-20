"""What happens when things go wrong, against a real control plane.

Failure is where a fake control plane is least like the real one — it accepts
whatever it's told. Two bugs got through that way: `apply` calling activate
unconditionally, which the real server rejects unless the workflow is paused or in
cooldown, and `resume` omitting the policy-decision id.
"""

from __future__ import annotations

import pytest
import yaml

from charter.config.loader import load_project
from charter.provisioning.apply import (
    AmbiguousInstance,
    apply_project,
    compile_agent,
    create_instance,
)
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_run
from tests.e2e.harness import calls, factory, scripted, submits
from tests.e2e.test_lifecycle import one_instance, project  # noqa: F401

pytestmark = pytest.mark.asyncio


async def run_one(cp, project, agent, wf, model, **context):
    worker = CharterWorker(project, chat_model=factory(model))
    async with running(worker):
        info = await wait_for_run(
            cp, await cp.invoke_workflow(wf.id, context=context), timeout=90)
    await worker.aclose()
    return info


async def test_apply_configures_without_activating_twice(cp, project, tenant):
    """A workflow is created paused and activate only accepts paused/cooldown, so
    calling it every time made the second apply fail. A fake control plane accepted
    it happily; this is the test that wouldn't have."""
    wf = await one_instance(cp, project, "ticket-sweeper", tenant)

    again = await apply_project(cp, project, only="ticket-sweeper", all_=True)
    assert [r.workflow_id for r in again] == [wf.id]


async def test_apply_refuses_to_guess_between_instances(cp, project, tenant):
    """Each instance has its own state, so configuring whichever came back first
    is the quiet kind of wrong."""
    bundle = project.agents["ticket-sweeper"]
    await create_instance(cp, compile_agent(bundle), tenant.id)
    await create_instance(cp, compile_agent(bundle), tenant.id)

    with pytest.raises(AmbiguousInstance):
        await apply_project(cp, project, only="ticket-sweeper")


@pytest.mark.xfail(reason="MCP tool failures aren't counted: the adapter returns "
                          "the error rather than raising, and the governed wrapper "
                          "counts failures on a raise. See charter/mcp/client.py.",
                   strict=True)
async def test_a_broken_tool_fails_the_task_naming_the_tool(cp, project, tenant):
    """`on_failure: fail` should end the task and name the integration to go look at.

    Marked xfail rather than deleted, because the behaviour is right and the
    plumbing isn't: nothing counts the failure, so nothing can act on it. Strict,
    so it fails loudly the day the wrapper learns to classify a returned error.
    """
    path = project.path.parent / "ticket-sweeper" / "v1.yaml"
    raw = yaml.safe_load(path.read_text())
    # Added, not substituted: runtime.yaml names the existing tools in its
    # tool_call_limits, and a limit on a tool no version declares is a config error
    # that quietly drops the agent from what the worker serves.
    raw["mcp"][0]["tools"].append({"tool": "always_fails", "on_failure": "fail"})
    raw["objective"] = "Call always_fails, then report."
    path.write_text(yaml.safe_dump(raw))
    reloaded = load_project(project.path)

    wf = await one_instance(cp, reloaded, "ticket-sweeper", tenant)
    model = scripted(
        calls("desk__always_fails", why="testing"),
        submits(summary="never gets here", needs_attention=0),
    )

    info = await run_one(cp, reloaded, "ticket-sweeper", wf, model)

    assert info.result.get("failed") is True, f"result was {info.result}"
    assert "always_fails" in info.result["reason"]


async def test_a_spent_budget_says_which_ceiling_it_hit(cp, project, tenant):
    """Not "it went round a lot" — the reason names the number that stopped it, so
    an operator knows whether to raise it or fix the agent."""
    path = project.path.parent / "ticket-sweeper" / "runtime.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["per_run"]["max_llm_calls"] = 2
    path.write_text(yaml.safe_dump(raw))
    reloaded = load_project(project.path)

    wf = await one_instance(cp, reloaded, "ticket-sweeper", tenant)
    # More turns than the cap allows and no submit, so the cap is the only way out.
    # (The script no longer repeats its last turn — a repeated tool call gets
    # replayed by subagents that may not have it, and loops.)
    model = scripted(*[calls("desk__list_open_tickets") for _ in range(6)])

    info = await run_one(cp, reloaded, "ticket-sweeper", wf, model)

    assert info.result["failed"] is True
    assert "max_llm_calls" in info.result["reason"]
    assert info.result["llm_calls"] > 0


async def test_a_failed_task_reports_how_far_it_got(cp, project, tenant):
    """The payload is what an operator reads when a rule pauses the agent, so it
    carries the spend rather than only the reason."""
    path = project.path.parent / "ticket-sweeper" / "runtime.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["per_run"]["max_llm_calls"] = 1
    path.write_text(yaml.safe_dump(raw))
    reloaded = load_project(project.path)

    wf = await one_instance(cp, reloaded, "ticket-sweeper", tenant)
    info = await run_one(cp, reloaded, "ticket-sweeper", wf,
                         scripted(*[calls("desk__list_open_tickets") for _ in range(6)]))

    assert set(info.result) >= {"failed", "reason", "cost_usd", "llm_calls", "gates"}
