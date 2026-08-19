"""YAML -> BoundFlow objects. Pure translation, no network.

The compiler's output *is* the SDK's own types, so a mismatch between what Charter
promises and what BoundFlow can execute is a type error here rather than a surprise
at apply time. `charter apply` is then just the sequence of calls that pushes these
objects at the control plane.

The reference implementation of that sequence is receiptbot's `provision()` — this
is the same thing with the constants read from files.
"""

from __future__ import annotations

from dataclasses import dataclass

from boundflow import (
    Cooldown,
    InvokeMode,
    Pause,
    CapabilityCallLimit,
    FileRule,
    RuntimePolicy,
    SetVersion,
    ToolCallLimit,
    WorkflowConfig,
    WorkflowMetric,
    WorkflowRule,
)

from .config.agent import AgentConfig
from .config.lifecycle import BOUNDFLOW_METRIC, LifecyclePolicyFile
from .config.loader import AgentBundle
from .config.runtime import RuntimePolicyFile


@dataclass
class CompiledAgent:
    """Everything `charter apply` needs to push for one agent version."""

    name: str          # workflow type, and the agent name policies are keyed by
    version: int
    workflow_config: WorkflowConfig
    runtime_policy: RuntimePolicy
    workflow_rules: list[WorkflowRule]

    @property
    def agent_name(self) -> str:
        """The name BoundFlow keys agent policy and metrics by.

        One Charter agent is one workflow containing one BoundFlow agent, so this is
        just the agent's name — which also makes traces and `invocation_metrics`
        read in Charter's vocabulary instead of an internal placeholder.
        """
        return self.name


def compile_workflow_config(cfg: AgentConfig, runtime: RuntimePolicyFile) -> WorkflowConfig:
    """`invoke_mode` is derived, never authored: an agent with inputs is doing
    discrete tasks and must queue, since coalescing would silently discard one.

    `invoke_timeout_seconds` is the entry operation's deadline; every later round
    gets the same number via Next.timeout. Left unset it defaults to 60s, so round
    one would be cancelled while rounds two onward had forty minutes.
    """
    return WorkflowConfig(
        version=cfg.version,
        invoke_mode=InvokeMode.QUEUE if cfg.invoke_mode == "queue" else InvokeMode.COALESCE,
        invoke_timeout_seconds=runtime.operation_timeout_seconds,
        repeat_every_seconds=cfg.schedule.every_seconds if cfg.schedule else 0,
        triggerable=cfg.schedule.manual if cfg.schedule else True,
        max_queue_depth=runtime.per_run.max_queue_depth,
    )


def compile_runtime_policy(cfg: AgentConfig, runtime: RuntimePolicyFile) -> RuntimePolicy:
    """Everything the worker enforces, from both files, in one policy.

    Two sources on purpose. Spend and counts are quantitative and live in
    `runtime.yaml`, which is re-applied every time and never rolls back. What the
    agent may *reach* — the capability allowlist and the file rules — is versioned
    config, so a `set_version` restores the authority that version ran with.
    They arrive together because the worker needs one object.

    Charter's per-run budget also becomes BoundFlow's per-invocation cap verbatim.
    Not a translation error: a gate ends the operation and the next one gets a
    fresh policy, so Charter passes what's left as a `Budget` while this stands as
    the in-worker backstop. Both defend the same declared number.

    `model` is left unset. The model lives in the versioned config and is handed to
    the harness directly; setting it here would be a second source of truth.
    """
    per_run = runtime.per_run
    return RuntimePolicy(
        max_llm_calls=per_run.max_llm_calls,
        max_cost_usd=per_run.max_cost_usd,
        max_tokens_per_call=runtime.limits.max_tokens_per_call,
        max_call_seconds=runtime.limits.max_call_seconds,
        tool_call_limits=[
            ToolCallLimit(tool=l.tool, max_calls=l.max_calls)
            for l in per_run.tool_call_limits
        ],
        # Declared here, enforced by the harness — which is the point. These are the
        # rules an engineer would otherwise hard-code where the agent is built;
        # policy arrives with the operation instead, and rolls back with it.
        capability_call_limits=[
            CapabilityCallLimit(capability=l.capability, max_calls=l.max_calls)
            for l in per_run.capability_call_limits
        ],
        file_rules=[
            FileRule(operations=list(r.operations), paths=list(r.paths), mode=r.mode)
            for r in cfg.file_rules
        ],
        allowed_capabilities=list(cfg.allowed_capabilities),
        # Declared MCP tools are always permitted, so the allowlist only ever has to
        # name what the *harness* brings. Empty means no allowlist, not "nothing".
        allowed_tools=cfg.all_tools() if cfg.allowed_capabilities else [],
    )
    # NOTE: max_tool_failures has no BoundFlow equivalent — it becomes a
    # tool_failure_limits Budget at the call itself, from task context.


def compile_workflow_rules(lifecycle: LifecyclePolicyFile | None) -> list[WorkflowRule]:
    """Charter exposes only workflow-level lifecycle policy, so this is the whole
    translation. `set_agent_lifecycle_policy` is never called, which is what lets
    Charter promise the effective runtime policy always equals the YAML."""
    if lifecycle is None:
        return []

    rules: list[WorkflowRule] = []
    for rule in lifecycle.rules:
        action = rule.then
        if action.pause:
            bf_action = Pause(window=action.pause.window)
        elif action.cooldown:
            bf_action = Cooldown(window=action.cooldown.window, seconds=action.cooldown.seconds)
        else:
            bf_action = SetVersion(target=action.set_version.target)

        rules.append(WorkflowRule(
            # `tool_failures` -> TOOL_FAILURE_RATE: a misnomer in the SDK, where the
            # engine compares a summed count. Charter uses the accurate name.
            metric=WorkflowMetric(BOUNDFLOW_METRIC[rule.when.metric]),
            threshold=rule.when.threshold,
            action=bf_action,
            tool=rule.when.tool,
        ))
    return rules


def compile_agent(bundle: AgentBundle, version: int | None = None) -> CompiledAgent:
    """Compile one version of an agent. Defaults to the newest on disk."""
    cfg = bundle.versions[version] if version is not None else bundle.latest
    return CompiledAgent(
        name=cfg.name,
        version=cfg.version,
        workflow_config=compile_workflow_config(cfg, bundle.runtime),
        runtime_policy=compile_runtime_policy(cfg, bundle.runtime),
        workflow_rules=compile_workflow_rules(bundle.lifecycle),
    )
