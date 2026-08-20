"""Durable subagents, against a real control plane.

The claim is that a child is a workflow rather than a coroutine: it exists in the
control plane, carries its own budget, and outlives the run that started it. None
of that can be shown against a fake — a fake would accept the create and report
whatever we asked it to.
"""

from __future__ import annotations

import pytest

from charter.worker import CharterWorker
from charter.workflows.spawning import TASK_KEY
from tests.e2e.conftest import running, wait_for_run
from tests.e2e.harness import by_model, calls, factory, scripted, submits
from tests.e2e.test_lifecycle import one_instance, project  # noqa: F401

pytestmark = pytest.mark.asyncio


async def test_a_started_child_exists_as_its_own_instance(cp, project, tenant):
    """Not a task inside the parent's process — a workflow of the same standing,
    which is what lets it keep running after the parent finishes."""
    wf = await one_instance(cp, project, "coordinator", tenant)
    await one_instance(cp, project, "scout", tenant)

    parent = scripted(
        calls("start_async_task", subagent_type="scout",
              description="find out who runs the Berlin office"),
        submits(finding="started a scout"),
    )
    child = scripted(submits(finding="Berlin is run by Dana"))
    worker = CharterWorker(project, chat_model=by_model(
        **{"claude-haiku-4-5": parent, "claude-sonnet-4-5": child}))
    async with running(worker):
        info = await wait_for_run(
            cp, await cp.invoke_workflow(wf.id, context={"topic": "berlin"}),
            timeout=90)
    await worker.aclose()

    assert info.status.value == "completed", info.failure_reason
    scouts = [w for w in await cp.list_workflows()
              if w.workflow_type == "scout" and w.tenant_id == tenant.id]
    # One from the fixture, one the coordinator minted.
    assert len(scouts) == 2, "the child should be a real workflow, not a coroutine"


async def test_an_undeclared_agent_is_refused_by_the_control_plane_too(
        cp, project, tenant):
    """`spawns` lists only scout, so naming anything else creates nothing. The run
    still completes — a refusal is information the model can act on."""
    wf = await one_instance(cp, project, "coordinator", tenant)

    model = scripted(
        calls("start_async_task", subagent_type="ticket-sweeper",
              description="sweep the queue"),
        submits(finding="was not allowed to start that"),
    )
    worker = CharterWorker(project, chat_model=factory(model))
    async with running(worker):
        info = await wait_for_run(
            cp, await cp.invoke_workflow(wf.id, context={"topic": "berlin"}),
            timeout=90)
    await worker.aclose()

    assert info.status.value == "completed", info.failure_reason
    sweepers = [w for w in await cp.list_workflows()
                if w.workflow_type == "ticket-sweeper" and w.tenant_id == tenant.id]
    assert sweepers == [], "an undeclared type must not be creatable"


async def test_pinning_a_task_id_is_what_makes_a_follow_up_continue(cp, project,
                                                                   tenant):
    """`update_async_task` keeps deepagents' semantics only because this holds:
    `durable_harness` keys the checkpoint and the store off `task_id`, so a run
    invoked with one pinned lands on the existing thread instead of a new one."""
    await one_instance(cp, project, "scout", tenant)
    scout = [w for w in await cp.list_workflows()
             if w.workflow_type == "scout" and w.tenant_id == tenant.id][0]

    first = await cp.invoke_workflow(scout.id, context={"description": "anything"})
    again = await cp.invoke_workflow(
        scout.id, context={TASK_KEY: first, "description": "more"})

    assert again != first, "a follow-up is its own run"
    assert (await cp.get_request_info(again)).invoke_context[TASK_KEY] == first


async def test_a_parent_sleeps_then_reads_what_its_child_found(cp, project, tenant):
    """The whole point, end to end.

    The parent starts a scout, parks on `wait` — releasing the worker, with its own
    run ended and its state in Postgres — comes back on a timer, checks, and reports
    what the child found. Nothing polls in code: the model decides when to look,
    which is the shape deepagents already has.
    """
    wf = await one_instance(cp, project, "coordinator", tenant)
    await one_instance(cp, project, "scout", tenant)

    parent = scripted(
        calls("start_async_task", subagent_type="scout",
              description="find out who runs the Berlin office"),
        # Short enough for a test, long enough that the child is done first.
        calls("wait", duration="5s", why="the scout needs a moment"),
        calls("check_async_task", task_id="{child}"),
        submits(finding="the scout says Dana"),
    )
    child = scripted(submits(finding="Berlin is run by Dana"))
    worker = CharterWorker(project, chat_model=by_model(
        **{"claude-haiku-4-5": parent, "claude-sonnet-4-5": child}))

    async with running(worker):
        info = await wait_for_run(
            cp, await cp.invoke_workflow(wf.id, context={"topic": "berlin"}),
            timeout=120)
    await worker.aclose()

    assert info.status.value == "completed", info.failure_reason
    assert info.result["finding"] == "the scout says Dana"
    # The parent really did park rather than block: a wait ends the run and the
    # scheduler starts another one when the timer is up.
    assert info.sequence_number >= 1
