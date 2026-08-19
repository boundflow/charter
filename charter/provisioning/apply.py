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

DEFAULT_TENANT = "default"


class AmbiguousInstance(Exception):
    """More than one instance and no choice made. Configuring the wrong entity is
    quiet and wrong, so the command stops rather than picking."""

    def __init__(self, agent: str, ids: list[str]) -> None:
        self.agent, self.ids = agent, ids
        super().__init__(agent)


class NoSuchTenant(Exception):
    """The named tenant doesn't exist. Deliberately not created on the fly: a typo
    would otherwise mint a second tenant with its own agents and its own history,
    and nothing would ever show you. `charter tenant create` is the only way one
    comes into existence."""

    def __init__(self, name: str, existing: list[str]) -> None:
        self.name, self.existing = name, existing
        super().__init__(name)


async def resolve_tenant(cp: ControlPlaneClient, name: str) -> str:
    tenants = await cp.list_tenants()
    for tenant in tenants:
        if tenant.name == name:
            return tenant.id
    raise NoSuchTenant(name, [t.name for t in tenants])


@dataclass
class ApplyResult:
    agent: str
    version: int
    workflow_id: str
    warnings: list[str] = field(default_factory=list)


async def instances_of(cp: ControlPlaneClient, compiled: CompiledAgent, tenant_id: str):
    """Every live instance of this agent, in creation order.

    Several workflows can share a type. They are instances of the same agent —
    each an entity with its own store namespace, its own budget, and its own
    lifecycle state — so this returns all of them rather than the first.

    Scoped by tenant, because an agent's identity is (tenant, name) and a
    workflow's tenant is fixed at creation. Matching on name alone would let
    `charter apply` against staging reconfigure production.
    """
    return [w for w in await cp.list_workflows()
            if (w.workflow_type == compiled.name and w.tenant_id == tenant_id
                and w.lifecycle_state not in _DEAD)]


async def create_instance(cp: ControlPlaneClient, compiled: CompiledAgent,
                          tenant_id: str):
    """Mint a new instance and configure it.

    Deliberately not part of `apply`. An instance is an entity with state of its
    own, so bringing one into existence is a decision someone makes, not a side
    effect of re-running config in CI.
    """
    workflow = await cp.create_workflow(
        compiled.name, tenant_id, config=compiled.workflow_config)
    await cp.set_agent_runtime_policy(
        workflow.id, compiled.agent_name, compiled.runtime_policy)
    await cp.set_workflow_lifecycle_policy(workflow.id, compiled.workflow_rules)
    # Created paused, and activate only accepts paused or cooldown — so it belongs
    # on the create path alone.
    await cp.activate_workflow(workflow.id)
    return workflow


async def apply_agent(
    cp: ControlPlaneClient,
    compiled: CompiledAgent,
    tenant_id: str,
    workflow,
    *,
    warnings: list[str] | None = None,
) -> ApplyResult:
    """Configure one existing instance. Never creates: see `create_instance`."""
    # The version is the only part of WorkflowConfig that moves on a re-apply.
    await cp.set_workflow_config(workflow.id, compiled.workflow_config)

    await cp.set_agent_runtime_policy(
        workflow.id, compiled.agent_name, compiled.runtime_policy)
    await cp.set_workflow_lifecycle_policy(workflow.id, compiled.workflow_rules)

    notes = list(warnings or [])
    state = getattr(workflow, "workflow_state", None)
    state = state.value if hasattr(state, "value") else state
    if state and state != "active":
        # Config is config: it applies whatever the agent is doing. But a rule
        # stopped this one for a reason, and a deploy silently restarting it is
        # the kind of surprise the rest of the design refuses. Fix the config,
        # then resume — two acts, because they're two decisions.
        notes.append(f"still {state} — new config applied, but no tasks will "
                     f"start until `charter resume {compiled.name}`")

    return ApplyResult(
        agent=compiled.name,
        version=compiled.version,
        workflow_id=workflow.id,
        warnings=notes,
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
            f"no runtime.yaml — using defaults "
            f"(max_cost_usd={per_run.max_cost_usd}, "
            f"max_tool_failures={per_run.max_tool_failures})")

    if cfg.gated_tools or cfg.file_rules_interrupt:
        if notifications is None or notifications.resolve(agent, "approval_requested") is None:
            out.append(
                "gates on approval but no notification route resolves — an approval "
                "nobody is told about is just a slow timeout")

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
    instances: list | None = None,
) -> list[ApplyResult]:
    """Apply one agent directory to `instances`, or to every live instance.

    No worker manifest required — the smallest path is a single v1.yaml plus
    BOUNDFLOW_* in the environment.
    """
    compiled = compile_agent(bundle, version)
    targets = instances or await instances_of(cp, compiled, tenant_id)
    warnings = _warnings(bundle, notifications)
    return [await apply_agent(cp, compiled, tenant_id, w, warnings=warnings)
            for w in targets]


async def apply_project(
    cp: ControlPlaneClient,
    project: Project,
    *,
    only: str | None = None,
    instance: str | None = None,
    all_: bool = False,
) -> list[ApplyResult]:
    """Apply every agent the manifest serves, to every live instance of each.

    `only` limits it to one agent. Nothing here creates an instance — an agent
    with none applies to nothing, and says so rather than quietly minting one.
    """
    cp_cfg = project.manifest.control_plane
    tenant_id = cp_cfg.tenant_id or await resolve_tenant(cp, cp_cfg.tenant)

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
        compiled = compile_agent(bundle, version)
        live = await instances_of(cp, compiled, tenant_id)
        if instance:
            live = [w for w in live if w.id.startswith(instance)]
        elif not all_ and len(live) > 1:
            raise AmbiguousInstance(served.agent, [w.id for w in live])
        results.extend(await apply_bundle(
            cp, bundle, tenant_id, version=version,
            notifications=project.manifest.notifications, instances=live))
    return results
