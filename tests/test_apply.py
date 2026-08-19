"""apply() against a fake control plane — the sequence and its idempotence, without
needing a server."""

import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from boundflow import LifecycleState

from charter.config.loader import load_project
from charter.compile import compile_agent
from charter.provisioning.apply import (
    AmbiguousInstance,
    apply_project,
    create_instance,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


@dataclass
class FakeWorkflow:
    id: str
    workflow_type: str
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE
    tenant_id: str = "tenant-1"
    workflow_state: str = "active"


@dataclass
class FakeTenant:
    id: str
    name: str


@dataclass
class FakeControlPlane:
    workflows: list[FakeWorkflow] = field(default_factory=list)
    calls: list[tuple] = field(default_factory=list)

    async def list_tenants(self):
        return [FakeTenant("tenant-1", "default")]

    async def list_workflows(self):
        return list(self.workflows)

    async def create_workflow(self, workflow_type, tenant_id, config=None):
        # The server creates a workflow paused; activate is what starts it. Ids are
        # unique per workflow — several can share a type, as instances of one agent.
        n = sum(1 for w in self.workflows if w.workflow_type == workflow_type)
        suffix = f"_{n}" if n else ""
        wf = FakeWorkflow(f"wf_{workflow_type}{suffix}", workflow_type, tenant_id=tenant_id,
                          workflow_state="paused")
        self.workflows.append(wf)
        self.calls.append(("create_workflow", workflow_type, tenant_id, config))
        return wf

    async def set_workflow_config(self, workflow_id, config):
        self.calls.append(("set_workflow_config", workflow_id, config))

    async def set_agent_runtime_policy(self, workflow_id, agent_name, policy):
        self.calls.append(("set_agent_runtime_policy", workflow_id, agent_name, policy))

    async def set_workflow_lifecycle_policy(self, workflow_id, rules):
        self.calls.append(("set_workflow_lifecycle_policy", workflow_id, rules))

    async def set_agent_lifecycle_policy(self, *a, **kw):  # pragma: no cover
        raise AssertionError("Charter must never set an agent lifecycle policy")

    async def activate_workflow(self, workflow_id):
        wf = next(w for w in self.workflows if w.id == workflow_id)
        if wf.workflow_state not in ("paused", "cooldown"):
            raise RuntimeError("workflow is not paused/cooldown")
        wf.workflow_state = "active"
        self.calls.append(("activate_workflow", workflow_id))

    async def set_model_pricing(self, model_id, input_per_1m, output_per_1m):
        self.calls.append(("set_model_pricing", model_id, input_per_1m, output_per_1m))

    def names(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def project(tmp_path):
    dst = tmp_path / "project"
    shutil.copytree(EXAMPLES, dst)
    return load_project(dst / "worker.yaml")


def run(coro):
    return asyncio.run(coro)


def instance(cp, project, agent="refund-triage"):
    """One live instance to configure. `apply` never makes these itself."""
    return run(create_instance(cp, compile_agent(project.agents[agent]), "tenant-1"))


def test_apply_never_creates(project):
    """An instance is an entity with its own state, so it comes into existence
    when someone decides that, not as a side effect of a config run in CI."""
    cp = FakeControlPlane()
    results = run(apply_project(cp, project))
    assert results == []
    assert "create_workflow" not in cp.names()


def test_apply_configures_the_instances_that_exist(project):
    cp = FakeControlPlane()
    instance(cp, project)
    cp.calls.clear()

    results = run(apply_project(cp, project, only="refund-triage"))
    assert [(r.agent, r.version) for r in results] == [("refund-triage", 1)]
    assert "create_workflow" not in cp.names()
    assert "set_workflow_config" in cp.names()
    assert len(cp.workflows) == 1


def test_creating_an_instance_configures_and_activates_it(project):
    cp = FakeControlPlane()
    instance(cp, project)
    assert cp.names() == [
        "create_workflow",
        "set_agent_runtime_policy",
        "set_workflow_lifecycle_policy",
        "activate_workflow",
    ]


def test_several_instances_without_a_choice_is_refused(project):
    """Each has its own state, so configuring whichever came back first is the
    quiet kind of wrong this is meant to prevent."""
    cp = FakeControlPlane()
    instance(cp, project)
    instance(cp, project)

    with pytest.raises(AmbiguousInstance) as e:
        run(apply_project(cp, project, only="refund-triage"))
    assert len(e.value.ids) == 2


def test_all_configures_every_instance(project):
    cp = FakeControlPlane()
    instance(cp, project)
    instance(cp, project)

    results = run(apply_project(cp, project, only="refund-triage", all_=True))
    assert len(results) == 2
    assert len({r.workflow_id for r in results}) == 2


def test_deleted_instances_are_not_configured(project):
    cp = FakeControlPlane([FakeWorkflow("wf_old", "refund-triage", LifecycleState.DELETED)])
    assert run(apply_project(cp, project, only="refund-triage")) == []


def test_policy_is_keyed_by_agent_name(project):
    cp = FakeControlPlane()
    instance(cp, project)
    run(apply_project(cp, project, only="refund-triage"))
    call = next(c for c in cp.calls if c[0] == "set_agent_runtime_policy")
    assert call[2] == "refund-triage"


def test_applies_the_newest_served_version_not_the_newest_on_disk(tmp_path):
    """ticket-summarizer has v1 and v2 on disk. A worker serving only v1 must get
    v1 applied — applying a version no worker holds strands the control plane."""
    dst = tmp_path / "project"
    shutil.copytree(EXAMPLES, dst)
    raw = yaml.safe_load((dst / "worker.yaml").read_text())
    for s in raw["serves"]:
        if s["agent"] == "ticket-summarizer":
            s["versions"] = [1]
    (dst / "worker.yaml").write_text(yaml.safe_dump(raw))

    cp = FakeControlPlane()
    reloaded = load_project(dst / "worker.yaml")
    instance(cp, reloaded, "ticket-summarizer")
    results = run(apply_project(cp, reloaded, only="ticket-summarizer"))
    assert results[0].version == 1


def test_warns_when_a_gated_agent_has_no_notification_route(tmp_path):
    """An approval nobody is told about is just a slow timeout."""
    dst = tmp_path / "project"
    shutil.copytree(EXAMPLES, dst)
    raw = yaml.safe_load((dst / "worker.yaml").read_text())
    raw.pop("notifications")
    (dst / "worker.yaml").write_text(yaml.safe_dump(raw))

    cp = FakeControlPlane()
    reloaded = load_project(dst / "worker.yaml")
    instance(cp, reloaded, "refund-triage")
    results = run(apply_project(cp, reloaded, only="refund-triage"))
    assert any("slow timeout" in w for w in results[0].warnings)


def test_no_warning_when_routed(project):
    cp = FakeControlPlane()
    instance(cp, project)
    results = run(apply_project(cp, project, only="refund-triage"))
    assert not any("slow timeout" in w for w in results[0].warnings)


def test_the_same_agent_name_in_another_tenant_is_a_different_agent(project):
    """A workflow's tenant is fixed at creation, so identity is (tenant, name).
    Matching on name alone would let `charter apply` against one tenant
    reconfigure another's agent."""
    cp = FakeControlPlane([FakeWorkflow("wf_other", "refund-triage",
                                        tenant_id="some-other-tenant")])
    mine = instance(cp, project)
    results = run(apply_project(cp, project, only="refund-triage"))
    assert [r.workflow_id for r in results] == [mine.id]
    assert len(cp.workflows) == 2


def test_reapply_does_not_activate_an_already_active_workflow(project):
    """A workflow is created paused and activate only accepts paused/cooldown, so
    calling it unconditionally made the second apply fail."""
    cp = FakeControlPlane()
    instance(cp, project)
    run(apply_project(cp, project, only="refund-triage"))
    cp.calls.clear()
    run(apply_project(cp, project, only="refund-triage"))
    assert "activate_workflow" not in cp.names()


def test_reapply_configures_a_paused_agent_without_resuming_it(project):
    """Config applies whatever the agent is doing — but a rule stopped this one for
    a reason, and restarting is a separate decision."""
    cp = FakeControlPlane([FakeWorkflow("wf_refund-triage", "refund-triage",
                                        workflow_state="paused")])
    results = run(apply_project(cp, project, only="refund-triage"))
    assert "set_agent_runtime_policy" in cp.names()
    assert "set_workflow_lifecycle_policy" in cp.names()
    assert "activate_workflow" not in cp.names()
    assert any("charter resume" in w for w in results[0].warnings)
