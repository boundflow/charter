"""One agent, end to end: apply, run, gate, approve, fail.

The model is a MockLlmClient scripted per test; the control plane, the MCP server
subprocess, the Charter loop and the gates are all real.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from boundflow import MockLlmClient, Turn
from boundflow.llm import SUBMIT_RESULT, ToolCall

from charter.config.loader import load_project
from charter.provisioning.apply import apply_project
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_gate, wait_for_run

pytestmark = pytest.mark.asyncio

PLAYGROUND = Path(__file__).parent.parent.parent / "playground"


@pytest.fixture
def project(tmp_path, tenant):
    """The playground, pointed at this test's own tenant."""
    dst = tmp_path / "playground"
    shutil.copytree(PLAYGROUND, dst)
    raw = yaml.safe_load((dst / "worker.yaml").read_text())
    raw["control_plane"]["tenant"] = tenant.name
    raw.pop("notifications", None)          # nothing is listening in a test
    (dst / "worker.yaml").write_text(yaml.safe_dump(raw))
    # The MCP server is spawned by relative path from the repo root.
    for agent in ("refund-demo", "ticket-sweeper"):
        v1 = dst / agent / "v1.yaml"
        cfg = v1.read_text().replace('args: ["playground/mcp_server.py"]',
                                     f'args: ["{PLAYGROUND / "mcp_server.py"}"]')
        v1.write_text(cfg)
    return load_project(dst / "worker.yaml")


def script(*turns):
    """A MockLlmClient that plays `turns` in order, recording what it was offered."""
    offered: list[list[str]] = []

    def next_turn(ctx):
        return turns[min(ctx.turn_index, len(turns) - 1)]

    client = MockLlmClient(next_turn)
    inner = client.complete

    async def complete(request):
        offered.append([t.name for t in request.tools])
        return await inner(request)

    client.complete = complete
    client.offered = offered
    return client


def submit(**fields):
    return Turn([ToolCall(SUBMIT_RESULT, fields)])


async def test_a_task_runs_and_publishes_its_deliverable(cp, project):
    await apply_project(cp, project, only="ticket-sweeper")
    llm = script(Turn([ToolCall("desk__list_open_tickets", {})]),
                 submit(summary="two tickets need a look", needs_attention=2))

    worker = CharterWorker(project, llm=llm)
    async with running(worker):
        wf = next(w for w in await cp.list_workflows()
                  if w.workflow_type == "ticket-sweeper")
        request_id = await cp.invoke_workflow(wf.id)
        info = await wait_for_run(cp, request_id)

    assert info.run_outcome.value == "successful"
    assert info.result["summary"] == "two tickets need a look"
    assert info.result["needs_attention"] == 2
    await worker.aclose()


async def test_the_model_is_never_offered_a_gated_tool(cp, project):
    """The central claim, checked against what actually went over the wire rather
    than against a stubbed ToolSet."""
    await apply_project(cp, project, only="refund-demo")
    llm = script(submit(resolution="no refund warranted", refunded_usd=0))

    worker = CharterWorker(project, llm=llm)
    async with running(worker):
        wf = next(w for w in await cp.list_workflows()
                  if w.workflow_type == "refund-demo")
        await wait_for_run(cp, await cp.invoke_workflow(wf.id, context={"ticket_id": "5150"}))

    assert llm.offered, "the model was never called"
    for tools in llm.offered:
        assert "desk__create_refund" not in tools
        assert "desk__get_ticket" in tools
    await worker.aclose()


async def test_a_proposal_parks_and_the_tool_runs_only_after_approval(cp, project):
    await apply_project(cp, project, only="refund-demo")
    llm = script(
        submit(propose={"tool": "desk__create_refund",
                        "args": {"charge_id": "ch_9002", "amount_usd": 240,
                                 "reason": "duplicate"},
                        "why": "ch_9001 and ch_9002 are the same charge"}),
        submit(resolution="refunded the duplicate", refunded_usd=240))

    worker = CharterWorker(project, llm=llm)
    async with running(worker):
        wf = next(w for w in await cp.list_workflows()
                  if w.workflow_type == "refund-demo")
        request_id = await cp.invoke_workflow(wf.id, context={"ticket_id": "4821"})

        gate = await wait_for_gate(cp, wf.id)
        assert "desk__create_refund" in gate.justification
        assert "240" in gate.justification

        await cp.approve_workflow(wf.id, gate.approval_id, "tester", "confirmed duplicate")
        info = await wait_for_run(cp, request_id)

    assert info.run_outcome.value == "successful"
    # An approved act is a step, not the end: the loop re-entered and finished.
    assert info.result["refunded_usd"] == 240
    assert info.result["acts_performed"][0]["tool"] == "desk__create_refund"
    await worker.aclose()


async def test_a_rejection_reaches_the_next_round(cp, project):
    """The only reason rejecting is worth more than cancelling."""
    await apply_project(cp, project, only="refund-demo")
    llm = script(
        submit(propose={"tool": "desk__create_refund",
                        "args": {"charge_id": "ch_9001", "amount_usd": 240,
                                 "reason": "duplicate"},
                        "why": "looks duplicated"}),
        submit(resolution="left it alone", refunded_usd=0))

    worker = CharterWorker(project, llm=llm)
    async with running(worker):
        wf = next(w for w in await cp.list_workflows()
                  if w.workflow_type == "refund-demo")
        request_id = await cp.invoke_workflow(wf.id, context={"ticket_id": "4821"})

        gate = await wait_for_gate(cp, wf.id)
        await cp.reject_workflow(wf.id, gate.approval_id, "tester",
                                 "wrong charge — ch_9001 is the original")
        info = await wait_for_run(cp, request_id)

    history = " ".join(info.result.get("history", []))
    assert "REJECTED" in history
    assert "wrong charge" in history
    await worker.aclose()
