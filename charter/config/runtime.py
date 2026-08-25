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

from .agent import AGENT_NAME, Base, Capability, FileRule


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
    # Working time, not wall clock: a task parked two days waiting for a human has
    # spent none of this. That is the useful reading — it bounds a runaway agent
    # rather than punishing a slow approver, and an agent that stops for you is the
    # behaviour we are trying to encourage.
    #
    # A deadline ("resolve this within the hour, human included") is a different
    # feature and wants a different name; it isn't this.
    max_seconds: float = Field(default=0.0, ge=0.0, description=(
        "Total time the agent may spend working on one task, across every round. "
        "0 is unlimited. Time parked waiting for a human doesn't count."))
    tool_call_limits: list[ToolCallLimit] = Field(default_factory=list)
    capability_call_limits: list[CapabilityLimit] = Field(default_factory=list)

    # Subagents get their own fields rather than being a capability limit. The
    # agent spawns one by calling a tool, but that's the harness's mechanism, and
    # writing `capability: spawn, max_calls: 5` to say "at most five subagents"
    # leaks it into the thing people read.
    #
    # Two numbers because they stop different things. The total is a budget: an
    # agent looping on subagents runs out of money eventually, but failing at a
    # stated ceiling says why. The parallel bound is a valve against a burst —
    # fifty spawned in one turn are all in flight before any has recorded a cost,
    # which no spend cap can catch in time.
    max_total_subagents: int = Field(default=0, ge=0, description=(
        "Subagents one task may spawn in total. 0 is unlimited."))
    max_parallel_subagents: int = Field(default=0, ge=0, description=(
        "Subagents that may be running at once. 0 is unlimited. Bounds only "
        "subagents — ordinary parallel tool calls are untouched."))

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
    # The other side of the same boundary. A model call is bounded by
    # max_call_seconds; a tool call had nothing — the MCP adapter offers a timeout
    # for HTTP servers and none at all for stdio, so a hung server blocked until
    # the control plane killed the whole round. That works and it is the bluntest
    # instrument available: a dead round with no explanation, rather than one tool
    # failing in a way the agent can read and work around.
    max_tool_seconds: float = Field(default=30.0, gt=0, description=(
        "How long one tool call may take before it's treated as a failure."))


# Floor so a tiny budget still gets room to dispatch; ceiling so a wedged round
# can't hold its lease for hours.
MIN_OPERATION_SECONDS, MAX_OPERATION_SECONDS = 60, 3600


class Authority(Base):
    """What the agent may reach, as opposed to how much it may spend.

    Not versioned, and that is the point: tightening what an agent can touch
    should not require cutting a release, and an approval signed in March should
    be checkable against the authority in force then — which the control plane
    snapshots per run — rather than against a number sealed into an artifact.
    """

    # Default-deny over the harness's own tools. Empty means no allowlist rather
    # than "nothing allowed": a field that silently forbade everything the moment
    # someone added it would be a bad default. Declared MCP tools are always
    # permitted regardless.
    allowed_capabilities: list[Capability] = Field(default_factory=list, description=(
        "read | write | execute | spawn. Empty means unrestricted."))
    file_rules: list[FileRule] = Field(default_factory=list)

    # Which agents this one may hand long-running work to. An allowlist, like the
    # capabilities above, and policy for the same reason: a child is a governed
    # unit with a budget of its own, and deciding who may mint one should not wait
    # for a release. Empty forbids — unlike `allowed_capabilities`, silence here
    # denies rather than permits, because the risk runs the other way.
    allowed_spawns: list[str] = Field(default_factory=list, description=(
        "Agent names this one may start as durable background tasks."))

    # How long a person has. Operational knobs: an approver's window is a fact
    # about your team, not about what the agent does, and shortening it should not
    # mint a version.
    approval_timeout_seconds: int = Field(default=1800, gt=0)
    question_timeout_seconds: int = Field(default=1800, gt=0, description=(
        "For `ask_human`. Whether the agent *can* ask stays versioned — that "
        "changes its tool list — but how long it waits is yours to tune."))
    max_wait_seconds: int = Field(default=0, ge=0, description=(
        "Ceiling on a single `wait`. 0 leaves the version's own `wait.max` as the "
        "only bound."))

    @model_validator(mode="after")
    def _check_spawns(self) -> Authority:
        if len(set(self.allowed_spawns)) != len(self.allowed_spawns):
            raise ValueError("`allowed_spawns` lists the same agent twice")
        return self


class RuntimePolicyFile(Base):
    apiVersion: Literal["charter/v1"]
    kind: Literal["RuntimePolicy"]

    agent: str = Field(pattern=AGENT_NAME)
    per_run: PerRun
    limits: Limits = Field(default_factory=Limits)
    authority: Authority = Field(default_factory=Authority)

    @model_validator(mode="after")
    def _check_authority(self) -> RuntimePolicyFile:
        """An agent that may spawn itself recurses with no bound.

        Each child gets a fresh budget, so nothing downstream stops it — this file
        is the only place it can be caught, and only here, because Authority alone
        does not know whose policy it is.
        """
        if self.agent in self.authority.allowed_spawns:
            raise ValueError(
                f"`allowed_spawns` includes {self.agent!r} itself. Each child gets "
                f"its own budget, so a self-spawning agent has nothing to stop it.")
        return self

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
        if self.per_run.max_seconds:
            # No round may outlast the whole task's allowance, or the ceiling would
            # only ever be noticed after it had already been passed.
            worst_case = min(worst_case, self.per_run.max_seconds)
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
