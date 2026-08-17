"""Pushing compiled agents at the control plane.

Everything here is idempotent — safe to run on every deploy, every boot, every CI
merge. That property comes from receiptbot's `provision()`, which this is a
generalization of: find-or-create, set policy unconditionally, activate.

Charter never calls `set_agent_lifecycle_policy`. That omission is a feature: with
no agent-lifecycle rules, the effective runtime policy always equals the base
policy, which is exactly what `runtime.yaml` says. No fired rule ever silently
changes a declared cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from boundflow import ControlPlaneClient, LifecycleState

from ..compile import CompiledAgent, compile_agent
from ..config.loader import AgentBundle, Project
from ..config.worker import Notifications

# A workflow in one of these states is gone; don't reuse it.
_DEAD = (LifecycleState.DELETED,)


@dataclass
class ApplyResult:
    agent: str
    version: int
    workflow_id: str
    created: bool
    warnings: list[str] = field(default_factory=list)


async def _find_or_create(cp: ControlPlaneClient, compiled: CompiledAgent, tenant_id: str):
    """Reuse a live workflow of this type rather than minting a new one — a fresh
    workflow has no run history, which would reset every lifecycle rule's window and
    throw away the metrics an operator is judging the agent by."""
    for w in await cp.list_workflows():
        if w.workflow_type == compiled.name and w.lifecycle_state not in _DEAD:
            return w, False
    return await cp.create_workflow(
        compiled.name, tenant_id, config=compiled.workflow_config), True


async def apply_agent(
    cp: ControlPlaneClient,
    compiled: CompiledAgent,
    tenant_id: str,
    *,
    warnings: list[str] | None = None,
) -> ApplyResult:
    workflow, created = await _find_or_create(cp, compiled, tenant_id)

    if not created:
        # The version is the only part of WorkflowConfig that moves on a re-apply.
        await cp.set_workflow_config(workflow.id, compiled.workflow_config)

    await cp.set_agent_runtime_policy(
        workflow.id, compiled.agent_name, compiled.runtime_policy)
    await cp.set_workflow_lifecycle_policy(workflow.id, compiled.workflow_rules)
    await cp.activate_workflow(workflow.id)

    return ApplyResult(
        agent=compiled.name,
        version=compiled.version,
        workflow_id=workflow.id,
        created=created,
        warnings=list(warnings or []),
    )


def _warnings(bundle: AgentBundle, notifications: Notifications | None) -> list[str]:
    """Things that are legal but almost certainly unintended. Warnings, not errors —
    a first agent shouldn't be blocked on operational polish."""
    out: list[str] = []
    cfg = bundle.latest
    agent = bundle.name

    if bundle.runtime_defaulted:
        per_run = bundle.runtime.per_run
        out.append(
            f"no runtime.yaml — using defaults (max_cost_usd={per_run.max_cost_usd}, "
            f"max_drafts={per_run.max_drafts})")

    if cfg.gated_tools or cfg.outcome.deliverable_approval == "always":
        if notifications is None or notifications.resolve(agent, "approval_requested") is None:
            out.append(
                "gates on approval but no notification route resolves — an approval "
                "nobody is told about is just a slow timeout")

    if cfg.outcome.ask_human is not None:
        if notifications is None or notifications.resolve(agent, "input_requested") is None:
            out.append("can ask a human but no notification route resolves for input_requested")

    # A server we connect to but barely use still puts every declared tool's schema
    # in the prompt on every call.
    for server in cfg.mcp:
        if len(server.tools) == 1:
            out.append(
                f"mcp server {server.name!r} is connected for a single tool — its "
                "schema rides along on every LLM call")

    return out


async def apply_bundle(
    cp: ControlPlaneClient,
    bundle: AgentBundle,
    tenant_id: str,
    *,
    version: int | None = None,
    notifications: Notifications | None = None,
) -> ApplyResult:
    """Apply one agent directory. No worker manifest required — the smallest path is
    a single v1.yaml plus BOUNDFLOW_* in the environment."""
    return await apply_agent(
        cp, compile_agent(bundle, version), tenant_id,
        warnings=_warnings(bundle, notifications))


async def apply_project(
    cp: ControlPlaneClient,
    project: Project,
    *,
    only: str | None = None,
) -> list[ApplyResult]:
    """Apply every agent the manifest serves. `only` limits it to one agent."""
    tenant_id = project.manifest.control_plane.tenant_id

    # Pricing is tenant-global in BoundFlow, not per-agent — which is why it lives
    # on the worker manifest and is applied once here.
    for model_id, price in project.manifest.model_pricing.items():
        await cp.set_model_pricing(model_id, input_per_1m=price.input, output_per_1m=price.output)

    results: list[ApplyResult] = []
    for served in project.manifest.serves:
        if only and served.agent != only:
            continue
        bundle = project.agents[served.agent]
        # Apply the newest version the manifest actually serves — applying a version
        # no worker holds would strand the control plane.
        version = max(v for v in served.versions if v in bundle.versions)
        results.append(await apply_bundle(
            cp, bundle, tenant_id, version=version,
            notifications=project.manifest.notifications))
    return results
