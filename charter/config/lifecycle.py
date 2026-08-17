"""The LifecyclePolicy file — `agents/<name>/lifecycle.yaml`.

Not versioned; always live. Workflow-level only — Charter never sets an agent
lifecycle policy, because the cap-adjusting actions would mutate BoundFlow's copy
of a budget without telling Charter's accumulator, and a model swap would change
behavior without a version bump, leaving the running agent described by no version
on disk.

Every action acts on the agent as a managed unit: hold it, slow it, or move it to a
known version. Those are the operations you want when thirty agents are running and
one is misbehaving.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .agent import AGENT_NAME, Base

# Charter's vocabulary. All map to BoundFlow's WorkflowMetric verbatim except
# `tool_failures`, which compiles to TOOL_FAILURE_RATE — a misnomer in the SDK,
# where the lifecycle engine reads ToolFailureCounts[tool] and compares a summed
# COUNT, never a ratio. Charter uses the accurate name and translates at compile
# time, so `threshold: 3` means three failed calls.
Metric = Literal[
    "num_failures",
    "cost",
    "num_llm_calls",
    "latency",
    "approval_rejections",
    "tool_failures",
]

BOUNDFLOW_METRIC = {
    "num_failures": "num_failures",
    "cost": "cost",
    "num_llm_calls": "num_llm_calls",
    "latency": "latency",
    "approval_rejections": "approval_rejections",
    "tool_failures": "tool_failure_rate",
}


class When(Base):
    metric: Metric
    # Summed over the action's window, compared >=.
    threshold: float
    # Namespaced; checked against the agent config by the loader.
    tool: str | None = None

    @model_validator(mode="after")
    def _check(self) -> When:
        if self.metric == "tool_failures" and not self.tool:
            raise ValueError("metric `tool_failures` requires `tool`")
        if self.metric != "tool_failures" and self.tool:
            raise ValueError(f"`tool` is only valid with metric `tool_failures`, not {self.metric!r}")
        return self


class Pause(Base):
    """Hold all new tasks until `charter resume`. Queued tasks wait, not discarded."""

    window: int = Field(gt=0)


class Cooldown(Base):
    """Pause, then auto-resume after `seconds`."""

    window: int = Field(gt=0)
    seconds: int = Field(gt=0)


class SetVersion(Base):
    """Roll to a known version. The target must exist on disk AND be served by every
    worker running this agent — the loader checks both."""

    target: int = Field(ge=1)


class Then(Base):
    pause: Pause | None = None
    cooldown: Cooldown | None = None
    set_version: SetVersion | None = None

    @model_validator(mode="after")
    def _check(self) -> Then:
        set_ = [n for n in ("pause", "cooldown", "set_version") if getattr(self, n)]
        if len(set_) != 1:
            raise ValueError(
                f"`then` takes exactly one of pause / cooldown / set_version, got "
                f"{set_ or 'none'}")
        return self

    @property
    def action(self) -> Pause | Cooldown | SetVersion:
        return self.pause or self.cooldown or self.set_version  # type: ignore[return-value]


class Rule(Base):
    when: When
    then: Then


class LifecyclePolicyFile(Base):
    apiVersion: Literal["charter/v1"]
    kind: Literal["LifecyclePolicy"]

    agent: str = Field(pattern=AGENT_NAME)
    rules: list[Rule] = Field(min_length=1)

    @property
    def version_targets(self) -> set[int]:
        """Versions any rule can roll this agent to."""
        return {r.then.set_version.target for r in self.rules if r.then.set_version}
