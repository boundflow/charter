"""What happens when things go wrong, against a real control plane.

Each of these is a failure Charter is supposed to name rather than crash on — and
the assertion is on the *reason*, because the reason is what an operator reads
after a lifecycle rule pauses the agent.
"""
from __future__ import annotations

import pytest
import yaml
from boundflow import MockLlmClient, Turn
from boundflow.llm import SUBMIT_RESULT, ToolCall

from charter.config.loader import load_project
from charter.provisioning.apply import apply_project
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_gate, wait_for_run
from tests.e2e.test_lifecycle import project, script, submit  # noqa: F401

pytestmark = pytest.mark.asyncio


async def _run(cp, project, agent, llm, **context):
    worker = CharterWorker(project, llm=llm)
    async with running(worker):
        wf = next(w for w in await cp.list_workflows() if w.workflow_type == agent)
        info = await wait_for_run(cp, await cp.invoke_workflow(wf.id, context=context))
    await worker.aclose()
    return info


async def test_apply_is_idempotent(cp, project):
    """A workflow is created paused and activate only accepts paused/cooldown, so
    calling it every time made the second apply fail. A fake control plane accepted
    it happily; this is the test that wouldn't have."""
    first = await apply_project(cp, project, only="ticket-sweeper")
    second = await apply_project(cp, project, only="ticket-sweeper")
    assert first[0].created is True
    assert second[0].created is False
    assert second[0].workflow_id == first[0].workflow_id


async def test_a_broken_tool_fails_the_task_naming_the_tool(cp, project, tmp_path):
    """on_failure: fail, and the reason says which integration to go look at."""
    raw = yaml.safe_load((project.path.parent / "ticket-sweeper" / "v1.yaml").read_text())
    raw["mcp"][0]["tools"] = [{"tool": "always_fails", "on_failure": "fail"}]
    raw["objective"] = "Call always_fails."
    (project.path.parent / "ticket-sweeper" / "v1.yaml").write_text(yaml.safe_dump(raw))
    reloaded = load_project(project.path)

    await apply_project(cp, reloaded, only="ticket-sweeper")

    llm = script(Turn([ToolCall("desk__always_fails", {"why": "testing"})]),
                 submit(summary="never gets here", needs_attention=0))
    info = await _run(cp, reloaded, "ticket-sweeper", llm)

    assert info.result["failed"] is True
    assert "desk__always_fails" in info.result["reason"]
    assert "on_failure" in info.result["reason"]


async def test_running_out_of_drafts_says_the_objective_is_wrong(cp, project):
    """Not 'it went round a lot' — the reason names what to go fix."""
    await apply_project(cp, project, only="refund-demo")
    proposal = submit(propose={"tool": "desk__create_refund",
                               "args": {"charge_id": "ch_9002", "amount_usd": 240,
                                        "reason": "duplicate"},
                               "why": "still think so"})
    llm = script(proposal, proposal, proposal, proposal)

    worker = CharterWorker(project, llm=llm)
    async with running(worker):
        wf = next(w for w in await cp.list_workflows() if w.workflow_type == "refund-demo")
        request_id = await cp.invoke_workflow(wf.id, context={"ticket_id": "4821"})

        # max_drafts is 2 in the playground, so the third rejection ends it.
        for _ in range(3):
            try:
                gate = await wait_for_gate(cp, wf.id, timeout=20)
            except AssertionError:
                break
            await cp.reject_workflow(wf.id, gate.approval_id, "tester", "no")
        info = await wait_for_run(cp, request_id)
    await worker.aclose()

    assert info.result["failed"] is True
    assert "rejected" in info.result["reason"]
    assert "objective or the agent is wrong" in info.result["reason"]
