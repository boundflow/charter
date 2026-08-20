"""Durable subagents.

The interface is deepagents', so the thing most worth pinning is that it stays
theirs — an agent written against deepagents-on-Platform is supposed to run here
unchanged, and a renamed tool or a renamed argument breaks that silently.
"""

from __future__ import annotations

import pytest

from pathlib import Path

from charter.config.loader import load_agent
from charter.workflows.spawning import Spawner, spawn_tools

EXAMPLE = Path(__file__).parent.parent / "examples" / "refund-triage"

pytestmark = pytest.mark.asyncio


class FakeWorkflow:
    def __init__(self, id_: str) -> None:
        self.id = id_


class FakeInfo:
    def __init__(self, status: str, result=None, workflow_id="wf-1") -> None:
        self.status = status
        self.result = result
        self.workflow_id = workflow_id


class FakeCP:
    """Records what it was told to do. Deliberately permissive — the point is what
    the spawner asks for, not what a control plane would say back."""

    def __init__(self, info: FakeInfo | None = None) -> None:
        self.created: list = []
        self.invoked: list = []
        self.deleted: list = []
        self._info = info

    async def create_workflow(self, name, tenant_id, config=None):
        self.created.append((name, tenant_id))
        return FakeWorkflow("wf-1")

    async def set_agent_runtime_policy(self, *a, **k): ...
    async def set_workflow_lifecycle_policy(self, *a, **k): ...
    async def activate_workflow(self, *a, **k): ...

    async def invoke_workflow(self, workflow_id, *, context=None, **k):
        self.invoked.append((workflow_id, context))
        return "task-abc"

    async def get_request_info(self, request_id):
        if self._info is None:
            raise RuntimeError("no such request")
        return self._info

    async def delete_workflow(self, workflow_id):
        self.deleted.append(workflow_id)


def said(result) -> str:
    """What the model is told, whether the tool answered with a string or a state
    update carrying a ToolMessage."""
    if isinstance(result, str):
        return result
    return str(result.update["messages"][0].content)


def recorded(result) -> dict:
    """The `async_tasks` entries a tool wrote into their channel."""
    return result.update.get("async_tasks", {})


CALL_ID = "call_1"


def state(*entries) -> dict:
    """The `async_tasks` channel a parent would be resuming with."""
    return {"async_tasks": {e["task_id"]: e for e in entries}}


def task(task_id="task-abc", workflow="wf-1", agent="scout", run=None) -> dict:
    return {"task_id": task_id, "agent_name": agent, "thread_id": workflow,
            "run_id": run or task_id, "status": "running"}


def bundles(project):
    return project.agents


# ── what may be spawned ─────────────────────────────────────────────────────


async def test_an_undeclared_agent_is_refused():
    """Refused as a tool result, not an exception: the model chose a name it
    wasn't offered, which is a mistake it can recover from by picking another."""
    cp = FakeCP()
    spawner = Spawner(cp, "t-1", {}, allowed=["outreach"])

    answer = await spawner.start("payroll", "do a thing", CALL_ID)

    assert "unknown subagent type" in answer, "deepagents' own wording"
    assert "outreach" in answer, "the refusal should say what is available"
    assert cp.created == [], "nothing should have been created"


async def test_a_declared_but_unserved_agent_says_so():
    """Distinct from the refusal above — this one is the operator's problem, and
    reads differently so nobody goes looking in the wrong config."""
    spawner = Spawner(FakeCP(), "t-1", {}, allowed=["outreach"])

    answer = await spawner.start("outreach", "do a thing", CALL_ID)

    assert "doesn't serve" in answer


# ── the four tools ──────────────────────────────────────────────────────────


async def test_the_tool_surface_is_deepagents_own():
    """The parity claim, asserted. Rename one of these and an agent written for
    deepagents stops working here for no reason a user could see."""
    names = {t.name for t in spawn_tools(Spawner(FakeCP(), "t", {}, []))}
    assert names == {"start_async_task", "check_async_task",
                     "update_async_task", "cancel_async_task"}


async def test_the_arguments_are_deepagents_own_too():
    """Imported from deepagents rather than restated, so this checks the import
    still resolves to what our functions actually take — a field they rename
    should break here, not in front of a model."""
    tools = {t.name: t for t in spawn_tools(Spawner(FakeCP(), "t", {}, []))}
    # The model-facing schema, not the full one — injected fields are added by us
    # and filtered out again before the model ever sees them.
    assert set(tools["start_async_task"].tool_call_schema.model_fields) == {
        "subagent_type", "description"}
    assert set(tools["update_async_task"].tool_call_schema.model_fields) == {
        "task_id", "message"}


async def test_the_start_tool_lists_what_may_be_spawned():
    """The model can only pick from `spawns`, so the list belongs in front of it —
    otherwise its first attempt is a guess that gets refused."""
    tools = {t.name: t for t in
             spawn_tools(Spawner(FakeCP(), "t", {}, ["scout", "outreach"]))}

    described = tools["start_async_task"].description
    assert "scout" in described and "outreach" in described


async def test_an_agent_that_declares_no_spawns_is_offered_nothing():
    """Not an empty-handed tool that refuses — the tool isn't there at all, so the
    model never proposes delegating and never spends a turn being told no."""
    assert spawn_tools(None) == []


# ── checking ────────────────────────────────────────────────────────────────


async def test_a_running_child_reports_no_result():
    spawner = Spawner(FakeCP(FakeInfo("running")), "t-1", {}, [], )

    answer = await spawner.check("task-abc", state(task("task-abc")))

    assert "running" in answer and "No result yet" in answer


async def test_a_finished_child_hands_back_its_result():
    info = FakeInfo("completed", result={"leads": 3})
    spawner = Spawner(FakeCP(info), "t-1", {}, [], )

    answer = await spawner.check("task-abc", state(task("task-abc")))

    assert "completed" in answer and "leads" in answer


async def test_a_lookup_failure_is_reported_not_raised():
    """A dead control plane shouldn't kill the parent's run — it's information the
    model can act on, like any other tool that couldn't answer."""
    answer = await Spawner(FakeCP(None), "t-1", {}, [], ).check("task-abc", state(task("task-abc")))

    assert "Failed to get run status" in answer, "deepagents' own wording"


# ── follow-ups ──────────────────────────────────────────────────────────────


async def test_cancelling_destroys_the_instance():
    cp = FakeCP(FakeInfo("running"))

    answer = await Spawner(cp, "t-1", {}, [], ).cancel("task-abc")

    assert cp.deleted == ["wf-1"]
    assert "cancelled" in answer


# ── ownership ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("act", ["check", "update", "cancel"])
async def test_a_task_this_parent_never_started_is_refused(act):
    """deepagents' check, inherited rather than reinvented: an id that isn't in
    this thread's `async_tasks` isn't actionable. It matters more here than there,
    because these ids address workflows that spend real money — guessing one must
    not let an agent cancel a sibling's work."""
    cp = FakeCP(FakeInfo("running"))
    spawner = Spawner(cp, "t-1", {}, [])
    mine = state(task("mine"))

    args = ["task-abc"] + (["a message"] if act == "update" else []) + [mine]
    if act == "update":
        args.append(CALL_ID)
    answer = said(await getattr(spawner, act)(*args))

    assert "no async task with id 'task-abc'" in answer
    assert "mine" in answer, "say which ids it does have"
    assert cp.deleted == [] and cp.invoked == []


async def test_a_follow_up_continues_the_same_thread():
    """The whole reason `update_async_task` can keep deepagents' semantics here:
    `durable_harness` keys the checkpoint and the store off `task_id`, so pinning
    it makes the child resume with everything it knew rather than start over."""
    cp = FakeCP(FakeInfo("running"))
    spawner = Spawner(cp, "t-1", {}, [])

    result = await spawner.update("task-abc", "also check Berlin",
                                  state(task("task-abc")), CALL_ID)

    workflow_id, context = cp.invoked[-1]
    assert workflow_id == "wf-1", "re-invokes the child's own instance"
    assert context["task_id"] == "task-abc", "pinned, so the thread continues"
    assert context["description"] == "also check Berlin"


async def test_a_follow_up_keeps_the_task_id_and_moves_the_run():
    """Their documented contract — the handle the parent holds stays put, and only
    the run behind it changes."""
    cp = FakeCP(FakeInfo("running"))
    spawner = Spawner(cp, "t-1", {}, [])

    written = recorded(await spawner.update(
        "task-abc", "more", state(task("task-abc")), CALL_ID))

    assert set(written) == {"task-abc"}, "no new handle appears"
    assert written["task-abc"]["run_id"] == "task-abc"


async def test_starting_a_child_writes_it_into_deepagents_own_channel():
    """Charter keeps no ledger. The record goes into `async_tasks`, which their
    reducer merges and the checkpointer persists per node — so the child is durably
    recorded the moment it exists, not whenever the operation ends."""
    spawner = Spawner(FakeCP(), "t-1", {"outreach": load_agent(EXAMPLE)},
                      ["outreach"])

    written = recorded(await spawner.start("outreach", "find leads", CALL_ID))

    assert written == {"task-abc": {
        "task_id": "task-abc", "agent_name": "outreach", "thread_id": "wf-1",
        "run_id": "task-abc", "status": "running"}}


async def test_cancelling_destroys_the_instance():
    cp = FakeCP(FakeInfo("running"))

    answer = await Spawner(cp, "t-1", {}, []).cancel(
        "task-abc", state(task("task-abc")))

    assert cp.deleted == ["wf-1"]
    assert "Cancelled task" in answer
