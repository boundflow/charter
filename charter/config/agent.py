"""The AgentConfig file — `agents/<name>/v<N>.yaml`.

The versioned artifact, and the only one that changes what an agent *does*. It is
immutable once applied: `charter apply` on an edited config writes v<N+1>.yaml,
because a `set_version` rollback must be able to rebuild exactly this agent.

Everything here validates in isolation. Rules that need a second file (does the
filename match `version`? does a lifecycle rule target a version that exists?) live
in the loader — see `charter.config.loader`.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Charter injects these branches into the agent's output schema; an author's own
# deliverable fields must not collide with them.
RESERVED_OUTPUT_FIELDS = frozenset({"propose", "ask_human"})

# The only templating Charter supports: {{ inputs.<name> }}, no expressions.
TEMPLATE_REF = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")

AGENT_NAME = r"^[a-z][a-z0-9-]{2,62}$"
SERVER_NAME = r"^[a-z][a-z0-9_]{1,31}$"
TOOL_NAME = r"^[a-z][a-z0-9_]{0,63}$"
ENV_NAME = r"^[A-Z][A-Z0-9_]*$"

ScalarType = Literal["string", "integer", "number", "boolean"]

_PYTYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


class Base(BaseModel):
    """Reject unknown keys everywhere. A typo in a safety-relevant file should fail
    loudly at apply time, not silently do nothing."""

    model_config = ConfigDict(extra="forbid")


def _matches(value: Any, type_: str) -> bool:
    # bool is an int subclass in Python; keep the two from bleeding into each other.
    if type_ == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    return isinstance(value, _PYTYPES[type_])


def template_refs(text: str) -> set[str]:
    """Every {{ ... }} reference in `text`, as written."""
    return {m.group(1) for m in TEMPLATE_REF.finditer(text)}


class InputSpec(Base):
    """One per-task input. Becomes a CLI flag on `charter run`, so scalars only —
    objects and arrays have no sane spelling as one."""

    type: ScalarType
    required: bool = False
    default: Any = None
    enum: list[Any] | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _check(self) -> InputSpec:
        if self.required and self.default is not None:
            raise ValueError("`required: true` and `default` are mutually exclusive")
        if self.enum is not None:
            if not self.enum:
                raise ValueError("`enum` must be non-empty")
            for v in self.enum:
                if not _matches(v, self.type):
                    raise ValueError(f"enum value {v!r} is not a {self.type}")
        if self.default is not None:
            if not _matches(self.default, self.type):
                raise ValueError(f"default {self.default!r} is not a {self.type}")
            if self.enum is not None and self.default not in self.enum:
                raise ValueError(f"default {self.default!r} is not in `enum`")
        return self


class ToolSpec(Base):
    """One tool on one MCP server.

    `approval` is binary on purpose: authorization is a decision a human makes in
    advance and writes down, never one delegated to the model's confidence in
    itself. Confidence only decides whether the agent asks for help.
    """

    tool: str = Field(pattern=TOOL_NAME)
    approval: Literal["never", "always"] = "never"
    on_failure: Literal["continue", "fail"] = "continue"

    @property
    def gated(self) -> bool:
        """True when the model never receives this tool and can only propose it."""
        return self.approval == "always"


class McpServer(Base):
    """One MCP server: a single endpoint or process exposing many tools.

    `tools` is a filter over what the server's `tools/list` returns. Tools the
    server exposes but this file does not declare are refused and never shown to
    the model, so a server shipping a new tool cannot silently widen the agent's
    reach. A declared tool the server does not expose quarantines the agent at boot.
    """

    name: str = Field(pattern=SERVER_NAME)
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    # Variable NAMES only. Values come from the worker's environment — this file is
    # committed and immutable, and must never carry a secret.
    env: list[str] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> McpServer:
        if bool(self.command) == bool(self.url):
            raise ValueError("exactly one of `command` or `url` is required")
        if self.args and not self.command:
            raise ValueError("`args` is only valid alongside `command`")
        if self.url and not self.url.startswith("https://"):
            raise ValueError("`url` must be https://")
        for name in self.env:
            if not re.match(ENV_NAME, name):
                raise ValueError(f"env {name!r} must be a variable NAME, not a value")
        seen = {t.tool for t in self.tools}
        if len(seen) != len(self.tools):
            raise ValueError("duplicate tool names")
        return self

    def qualified(self, tool: str) -> str:
        return f"{self.name}.{tool}"

    @property
    def tool_names(self) -> list[str]:
        return [self.qualified(t.tool) for t in self.tools]


class FieldSpec(Base):
    type: ScalarType
    description: str | None = None


class ApprovalSpec(Base):
    """How an approval gate behaves. Not what it says — the approver is always shown
    the proposed tool, its arguments, and the agent's own justification, and that
    rendering is Charter's. Only `note` is authored, so a gate cannot be written
    with the amount left out."""

    timeout_seconds: int = Field(default=1800, gt=0)
    on_timeout: Literal["reject", "fail"] = "reject"
    note: str | None = None


# How eager the agent should be to ask, as prompt language rather than a number.
# Charter owns the wording so it's consistent and testable, and so an author can't
# accidentally write something weaker than they meant.
ASK_POSTURE = {
    "rarely": (
        "Only ask a human when you are genuinely blocked — when you cannot make "
        "progress at all without an answer. Prefer investigating with your tools "
        "first. An unnecessary question wastes someone's time."),
    "balanced": (
        "Ask a human when something material is ambiguous and you cannot resolve it "
        "with your tools. Do not ask about things you can look up, and do not guess "
        "about things you cannot."),
    "eagerly": (
        "When in doubt, ask. Being wrong here is more costly than asking an extra "
        "question, so raise anything you are unsure about rather than deciding it "
        "yourself."),
}


class AskHumanSpec(Base):
    """The reasoning escape hatch: the agent is stuck, so it asks a person rather
    than guessing, folds the answer into history, and tries again.

    There is deliberately no `prompt` — the question comes from the agent at
    runtime, because you can't know in advance what it will get stuck on.

    `when` shapes a tendency through prompt language. There is no self-reported
    confidence score anywhere: a model's opinion of its own certainty is poorly
    calibrated — a confidently wrong answer reports high — so it isn't wired to
    anything. How many questions it may ask is a limit, and lives in runtime.yaml.
    """

    timeout_seconds: int = Field(default=240, gt=0)
    on_timeout: Literal["continue", "fail"] = "continue"
    # Prompt guidance. A tendency, not a guarantee. How MANY questions it may ask
    # is `per_run.max_questions` in runtime.yaml — a limit, not a behaviour, so it
    # belongs with the other limits.
    when: Literal["rarely", "balanced", "eagerly"] = "balanced"

    @property
    def guidance(self) -> str:
        return ASK_POSTURE[self.when]


class AuditMemory(Base):
    """Memory derived from the governance audit log — no store, no embeddings, no
    extraction pass. Every row is a human judgment about this specific agent, which
    is higher-signal than summarized conversation history and, unlike an embedding
    index, exactly inspectable: `charter memory <agent>` prints the text verbatim.

    Only covers memory of human judgments about the agent. Memory of the *world*
    (documents, customer history, semantic recall) is an MCP server with a read tool
    and a write tool, governed like any other.
    """

    # Recent rejected proposals and the reason given. Requires something to gate.
    rejections: int = Field(default=0, ge=0, le=50)
    # Recent answers a human gave to the agent's own questions. Requires ask_human.
    answers: int = Field(default=0, ge=0, le=50)


class Memory(Base):
    from_audit: AuditMemory | None = None


class Schedule(Base):
    """Run this agent on a clock instead of waiting to be asked.

    Mutually exclusive with `inputs`: a periodic run has nobody to supply a ticket
    id, so an agent that needs one can't be scheduled. That agent looks for its own
    work instead — which is why a scheduled agent coalesces: two ticks that both
    mean "something changed, go look" really are one piece of work.
    """

    # "15m", "1h", "30s", "7d" — seconds are what BoundFlow takes, but nobody
    # thinks in seconds past about a minute.
    every: str
    # Whether `charter run` still works. Left on by default so a scheduled agent
    # can be tested by hand without editing its config.
    manual: bool = True

    @property
    def every_seconds(self) -> int:
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        raw = self.every.strip()
        if raw[-1:] in units and raw[:-1].isdigit():
            return int(raw[:-1]) * units[raw[-1]]
        raise ValueError(f"every: {self.every!r} — use 30s, 15m, 1h, 7d")

    @model_validator(mode="after")
    def _check(self) -> Schedule:
        self.every_seconds  # raises with the readable message
        return self


class Outcome(Base):
    """The three ways a task can leave the agent loop.

    Only a deliverable ends a task. An approved act is a step: the tool runs, its
    result folds into history, and the loop re-enters — so the agent can act again
    and always gets to report what it did.
    """

    deliverable: dict[str, FieldSpec] = Field(min_length=1)
    deliverable_approval: Literal["never", "always"] = "never"
    approval: ApprovalSpec | None = None
    ask_human: AskHumanSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> Outcome:
        collisions = RESERVED_OUTPUT_FIELDS & self.deliverable.keys()
        if collisions:
            raise ValueError(
                f"deliverable fields {sorted(collisions)} are reserved — Charter "
                "injects them into the agent's output schema")
        return self


class AgentConfig(Base):
    apiVersion: Literal["charter/v1"]
    kind: Literal["AgentConfig"]

    name: str = Field(pattern=AGENT_NAME)
    version: int = Field(ge=1)
    description: str | None = None

    model: str = Field(min_length=1)
    objective: str = Field(min_length=1)

    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    schedule: Schedule | None = None
    mcp: list[McpServer] = Field(default_factory=list)
    outcome: Outcome
    memory: Memory | None = None

    @model_validator(mode="after")
    def _check(self) -> AgentConfig:
        self._check_servers_unique()
        self._check_templates()
        self._check_gates()
        self._check_memory()
        self._check_schedule()
        return self

    def _check_schedule(self) -> None:
        if self.schedule and self.inputs:
            raise ValueError(
                "`schedule` and `inputs` are mutually exclusive — a periodic run has "
                "nobody to supply them. A scheduled agent finds its own work.")

    def _check_servers_unique(self) -> None:
        names = [s.name for s in self.mcp]
        if len(set(names)) != len(names):
            raise ValueError("duplicate mcp server names")
        qualified = [t for s in self.mcp for t in s.tool_names]
        if len(set(qualified)) != len(qualified):
            raise ValueError("duplicate qualified tool names")

    def _check_templates(self) -> None:
        """Only {{ inputs.<name> }}, and only for inputs that exist. An undeclared
        reference is a mistake we can catch now instead of at task time."""
        sources = [("objective", self.objective)]
        if self.outcome.approval and self.outcome.approval.note:
            sources.append(("outcome.approval.note", self.outcome.approval.note))

        for where, text in sources:
            for ref in template_refs(text):
                if not ref.startswith("inputs."):
                    raise ValueError(
                        f"{where}: {{{{ {ref} }}}} — only {{{{ inputs.<name> }}}} is "
                        "supported")
                key = ref.split(".", 1)[1]
                if key not in self.inputs:
                    raise ValueError(f"{where}: {{{{ {ref} }}}} is not a declared input")

    def _check_gates(self) -> None:
        """An `approval` block is required exactly when something can gate, and
        forbidden otherwise — a mutating tool with no gate configured would be a
        silent hole, and a gate configured for an agent that never gates is dead
        config that reads as protection."""
        gates = bool(self.gated_tools) or self.outcome.deliverable_approval == "always"
        if gates and self.outcome.approval is None:
            raise ValueError(
                "outcome.approval is required: "
                + ("tools with `approval: always`" if self.gated_tools else "")
                + (" and " if self.gated_tools and self.outcome.deliverable_approval == "always" else "")
                + ("`deliverable_approval: always`" if self.outcome.deliverable_approval == "always" else "")
                + " need a gate to be configured")
        if not gates and self.outcome.approval is not None:
            raise ValueError(
                "outcome.approval is set but nothing gates — no tool declares "
                "`approval: always` and `deliverable_approval` is `never`")

    def _check_memory(self) -> None:
        """Audit memory can only recall what the audit log holds. An agent with no
        gates never accrues rejections; one that can't ask never accrues answers.
        Configuring either would read as memory the agent doesn't have."""
        audit = self.memory.from_audit if self.memory else None
        if audit is None:
            return
        gates = bool(self.gated_tools) or self.outcome.deliverable_approval == "always"
        if audit.rejections and not gates:
            raise ValueError(
                "memory.from_audit.rejections needs something to gate — no tool "
                "declares `approval: always` and `deliverable_approval` is `never`, "
                "so this agent is never rejected")
        if audit.answers and self.outcome.ask_human is None:
            raise ValueError(
                "memory.from_audit.answers needs `outcome.ask_human` — this agent "
                "cannot ask, so no answers are ever recorded")

    # ── Derived views the compiler and worker read ──────────────────────────

    @property
    def all_tools(self) -> list[str]:
        """Every declared tool, namespaced."""
        return [t for s in self.mcp for t in s.tool_names]

    @property
    def gated_tools(self) -> list[str]:
        """Tools the model never receives and can only propose."""
        return [s.qualified(t.tool) for s in self.mcp for t in s.tools if t.gated]

    @property
    def inline_tools(self) -> list[str]:
        """Tools handed to the model and called inside the loop."""
        return [s.qualified(t.tool) for s in self.mcp for t in s.tools if not t.gated]

    @property
    def fail_fast_tools(self) -> set[str]:
        """Tools whose failure fails the task at the next iteration boundary."""
        return {s.qualified(t.tool) for s in self.mcp for t in s.tools
                if t.on_failure == "fail"}

    @property
    def invoke_mode(self) -> Literal["queue", "coalesce"]:
        """Derived, never authored. An agent with inputs is doing discrete tasks and
        must queue — coalescing would silently discard a ticket. One with no inputs
        is being told "something changed, go look", where two triggers really are
        one piece of work."""
        return "queue" if self.inputs else "coalesce"
