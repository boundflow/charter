"""The RuntimePolicy file — `agents/<name>/runtime.yaml`.

Not versioned: re-applied on every `charter apply`, so a `set_version` rollback
restores old behavior while leaving today's guardrails in force.

Purely quantitative. Nothing here changes what the agent can reach or how it
behaves — that all lives in the versioned config. Which is what lets us promise the
effective policy always equals what this file says.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .agent import AGENT_NAME, Base, Capability


class ToolCallLimit(Base):
    # Namespaced <server>.<tool>; checked against the agent config by the loader.
    tool: str
    max_calls: int = Field(gt=0)


class CapabilityLimit(Base):
    """A cap on how many times the agent may do a *kind* of thing.

    Capping a tool doesn't hold when the harness ships three ways to do the same
    thing: cap `write_file` and the agent reaches for `edit_file`. `write` covers
    both, plus `delete`, and keeps covering whatever the next release adds.

    This is the only harness limit that lives here, because it's the only one that's
    purely quantitative. What the agent may *reach* — file rules, the capability
    allowlist — is versioned config, so it rolls back with the agent.
    """

    capability: Capability = Field(description=(
        "read (ls, read_file, glob, grep) | write (write_file, edit_file, delete) | "
        "execute (shell) | spawn (subagents)"))
    max_calls: int = Field(gt=0, description="Calls of this kind allowed in one task.")


class PerRun(Base):
    """Limits on one task — one `charter run`, however long the agent takes and
    however many times it stops for a human.

    Spend is the reason this file exists. Nothing in the harness knows what a dollar
    is, so `max_cost_usd` and `max_llm_calls` are the caps only Charter can enforce,
    and they bound the whole task rather than one step of it.

    The rest are declared here and enforced by the harness's own mechanisms, so they
    are versioned policy rather than something hard-coded wherever the agent was
    built. `max_tool_failures` is the exception that stays ours: a repeatedly failing
    integration has no harness equivalent, and it should trip its own breaker rather
    than quietly burning the budget.
    """

    max_cost_usd: float = Field(default=0.0, ge=0.0, description=(
        "Total spend for one task, across every round and every retry."))
    max_llm_calls: int = Field(default=0, ge=0, description=(
        "Total model calls for one task. One round makes several."))
    tool_call_limits: list[ToolCallLimit] = Field(default_factory=list)
    capability_call_limits: list[CapabilityLimit] = Field(default_factory=list)

    # Failures of any ONE tool before the task gives up — a circuit breaker, per
    # tool rather than in aggregate, so the failure message names the broken thing.
    max_tool_failures: int = Field(default=3, gt=0, description=(
        "Failures of any ONE tool before the task gives up — a circuit breaker, "
        "per tool rather than in aggregate."))
    # How many tasks may pile up unstarted before invoke is refused. Queue mode
    # only — coalesce keeps the latest and discards the rest by design. 0 uses
    # BoundFlow's default of 1000.
    max_queue_depth: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> PerRun:
        if not (self.max_cost_usd or self.max_llm_calls):
            raise ValueError(
                "at least one of max_cost_usd / max_llm_calls is required — an agent "
                "with no spend ceiling is a mistake, not a choice")
        seen = {l.tool for l in self.tool_call_limits}
        if len(seen) != len(self.tool_call_limits):
            raise ValueError("duplicate tool in tool_call_limits")
        caps = {l.capability for l in self.capability_call_limits}
        if len(caps) != len(self.capability_call_limits):
            raise ValueError("duplicate capability in capability_call_limits")
        return self


class Limits(Base):
    """Valves against one pathological call, not budgets. A per-task token budget
    can't save you from a single response that returns 200k tokens."""

    max_tokens_per_call: int = Field(default=1024, gt=0)
    max_call_seconds: float = Field(default=60.0, gt=0)


# Floor so a tiny budget still gets room to dispatch; ceiling so a wedged round
# can't hold its lease for hours.
MIN_OPERATION_SECONDS, MAX_OPERATION_SECONDS = 60, 3600


class RuntimePolicyFile(Base):
    apiVersion: Literal["charter/v1"]
    kind: Literal["RuntimePolicy"]

    agent: str = Field(pattern=AGENT_NAME)
    per_run: PerRun
    limits: Limits = Field(default_factory=Limits)

    @property
    def operation_timeout_seconds(self) -> int:
        """How long one round may take before the control plane cancels it.

        Derived, not picked: a round is one agent step — up to `max_llm_calls`
        calls, each bounded by `max_call_seconds`. A fixed value would cancel
        exactly the slow, tool-heavy rounds and leave the fast ones alone.

        Both ends of a task need the same number. The entry operation takes it from
        WorkflowConfig.invoke_timeout_seconds, every later round from Next.timeout —
        so this lives here rather than in either caller, where they could drift.
        """
        worst_case = (self.per_run.max_llm_calls or 20) * self.limits.max_call_seconds
        return int(min(max(worst_case, MIN_OPERATION_SECONDS), MAX_OPERATION_SECONDS))


# What an agent gets when it ships no runtime.yaml. Deliberately conservative: an
# agent must always have a ceiling, but requiring the file to get one is friction
# for someone whose first agent is a single YAML. `charter apply` prints these when
# they're used, so an unauthored budget is never a silent one.
DEFAULT_PER_RUN = PerRun(max_cost_usd=1.00)


def default_runtime(agent: str) -> RuntimePolicyFile:
    return RuntimePolicyFile(
        apiVersion="charter/v1",
        kind="RuntimePolicy",
        agent=agent,
        per_run=DEFAULT_PER_RUN.model_copy(deep=True),
    )
