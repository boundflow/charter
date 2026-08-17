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
from charter.provisioning.apply import apply_project

EXAMPLES = Path(__file__).parent.parent / "examples"


@dataclass
class FakeWorkflow:
    id: str
    workflow_type: str
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE
    tenant_id: str = "tenant-1"


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
        wf = FakeWorkflow(f"wf_{workflow_type}", workflow_type, tenant_id=tenant_id)
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


def test_creates_both_agents(project):
    cp = FakeControlPlane()
    results = run(apply_project(cp, project))
    assert [(r.agent, r.version, r.created) for r in results] == [
        ("refund-triage", 1, True),
        ("ticket-summarizer", 2, True),
    ]


def test_call_sequence_per_agent(project):
    cp = FakeControlPlane()
    run(apply_project(cp, project, only="refund-triage"))
    assert cp.names() == [
        "set_model_pricing", "set_model_pricing",
        "create_workflow",
        "set_agent_runtime_policy",
        "set_workflow_lifecycle_policy",
        "activate_workflow",
    ]


def test_reapply_reuses_the_workflow(project):
    """A fresh workflow has no run history, which would reset every lifecycle
    window and discard the metrics an operator is judging the agent by."""
    cp = FakeControlPlane()
    run(apply_project(cp, project, only="refund-triage"))
    cp.calls.clear()

    results = run(apply_project(cp, project, only="refund-triage"))
    assert results[0].created is False
    assert results[0].workflow_id == "wf_refund-triage"
    assert "create_workflow" not in cp.names()
    assert "set_workflow_config" in cp.names()
    assert len(cp.workflows) == 1


def test_deleted_workflow_is_not_reused(project):
    cp = FakeControlPlane([FakeWorkflow("wf_old", "refund-triage", LifecycleState.DELETED)])
    results = run(apply_project(cp, project, only="refund-triage"))
    assert results[0].created is True
    assert results[0].workflow_id == "wf_refund-triage"


def test_policy_is_keyed_by_agent_name(project):
    cp = FakeControlPlane()
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
    results = run(apply_project(
        cp, load_project(dst / "worker.yaml"), only="ticket-summarizer"))
    assert results[0].version == 1


def test_warns_when_a_gated_agent_has_no_notification_route(tmp_path):
    """An approval nobody is told about is just a slow timeout."""
    dst = tmp_path / "project"
    shutil.copytree(EXAMPLES, dst)
    raw = yaml.safe_load((dst / "worker.yaml").read_text())
    raw.pop("notifications")
    (dst / "worker.yaml").write_text(yaml.safe_dump(raw))

    cp = FakeControlPlane()
    results = run(apply_project(
        cp, load_project(dst / "worker.yaml"), only="refund-triage"))
    assert any("slow timeout" in w for w in results[0].warnings)


def test_no_warning_when_routed(project):
    cp = FakeControlPlane()
    results = run(apply_project(cp, project, only="refund-triage"))
    assert not any("slow timeout" in w for w in results[0].warnings)


def test_the_same_agent_name_in_another_tenant_is_a_different_agent(project):
    """A workflow's tenant is fixed at creation, so identity is (tenant, name).
    Matching on name alone would let `charter apply` against one tenant
    reconfigure another's agent."""
    cp = FakeControlPlane([FakeWorkflow("wf_other", "refund-triage",
                                        tenant_id="some-other-tenant")])
    results = run(apply_project(cp, project, only="refund-triage"))
    assert results[0].created is True
    assert results[0].workflow_id == "wf_refund-triage"
    assert len(cp.workflows) == 2
