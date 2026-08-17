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

from .agent import AGENT_NAME, Base


class ToolCallLimit(Base):
    # Namespaced <server>.<tool>; checked against the agent config by the loader.
    tool: str
    max_calls: int = Field(gt=0)


class PerRun(Base):
    """Limits on one task — one `charter run`, however much back-and-forth it takes.

    Two kinds. `max_cost_usd` and `max_llm_calls` are *spend*: Charter passes what's
    left to each agent step as a `Budget`, so BoundFlow's per-step cap can't reset
    its way past the declared total.

    The rest bound *not converging*, and each names a distinct cause, so the failure
    an operator reads says which one happened:

      max_drafts         you kept rejecting  -> the agent or the objective is wrong
      max_questions      it kept asking      -> it lacks the context to do this
      max_tool_failures  a tool kept failing -> the integration is broken

    There is deliberately no user-facing round or iteration count. "It went round six
    times" is an implementation detail that diagnoses nothing; these three each point
    at something you can act on.
    """

    max_cost_usd: float = Field(default=0.0, ge=0.0)
    max_llm_calls: int = Field(default=0, ge=0)
    tool_call_limits: list[ToolCallLimit] = Field(default_factory=list)

    # Submissions a human may reject before the task is given up on. An approved
    # act does not count — you accepted that one; the agent is drafting the next.
    max_drafts: int = Field(default=3, gt=0)
    # Questions the agent may ask before it must show you something. Resets after
    # each draft, so a rejection earns it a fresh allowance rather than a dead end.
    max_questions: int = Field(default=2, gt=0)
    # Failures of any ONE tool before the task gives up — a circuit breaker, per
    # tool rather than in aggregate, so the failure message names the broken thing.
    max_tool_failures: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def _check(self) -> PerRun:
        if not (self.max_cost_usd or self.max_llm_calls):
            raise ValueError(
                "at least one of max_cost_usd / max_llm_calls is required — an agent "
                "with no spend ceiling is a mistake, not a choice")
        seen = {l.tool for l in self.tool_call_limits}
        if len(seen) != len(self.tool_call_limits):
            raise ValueError("duplicate tool in tool_call_limits")
        return self


class Limits(Base):
    """Valves against one pathological call, not budgets. A per-task token budget
    can't save you from a single response that returns 200k tokens."""

    max_tokens_per_call: int = Field(default=1024, gt=0)
    max_call_seconds: float = Field(default=60.0, gt=0)


class RuntimePolicyFile(Base):
    apiVersion: Literal["charter/v1"]
    kind: Literal["RuntimePolicy"]

    agent: str = Field(pattern=AGENT_NAME)
    per_run: PerRun
    limits: Limits = Field(default_factory=Limits)


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
