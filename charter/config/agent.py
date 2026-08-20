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
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Charter injects these branches into the agent's output schema; an author's own
# Charter no longer injects fields of its own into the model's output, so there is
# nothing for a response_format field to collide with.
RESERVED_OUTPUT_FIELDS: frozenset[str] = frozenset()

# The only templating Charter supports: {{ inputs.<name> }}, no expressions.
TEMPLATE_REF = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")

AGENT_NAME = r"^[a-z][a-z0-9-]{2,62}$"
# What the model providers accept for a tool name. Enforced on the qualified name
# so a config can't produce a request that 400s.
WIRE_TOOL_NAME = r"^[a-zA-Z0-9_-]{1,128}$"
SERVER_NAME = r"^[a-z][a-z0-9_]{1,31}$"
# Hyphens allowed: real servers use them (`tavily-search`), and our own wire
# pattern always did. Rejecting them here meant a tool that exists couldn't be
# declared — the config was stricter than the protocol for no reason.
TOOL_NAME = r"^[a-z][a-z0-9_-]{0,63}$"
ENV_NAME = r"^[A-Z][A-Z0-9_]*$"
# How a tool is namespaced onto its server. Double underscore because a dot is
# rejected by the providers' tool-name charset — see McpServer.qualified.
SEPARATOR = "__"

ScalarType = Literal["string", "integer", "number", "boolean"]
FieldType = Literal["string", "integer", "number", "boolean", "array", "object"]

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
    # None means "unset": follow the server's `approval` rules if it has any, and
    # `never` if it doesn't. An explicit value always wins over a rule.
    approval: Literal["never", "always"] | None = Field(default=None, description=(
        "never: handed to the model, called inline. always: the model never "
        "receives it and can only propose it. Unset follows the server's "
        "approval rules, or never if it has none."))
    approval_timeout_seconds: int | None = Field(default=None, gt=0, description=(
        "How long an approver has for *this* tool, overriding the agent's `gate`. "
        "A refund can wait; a production deploy might not."))
    on_failure: Literal["continue", "fail"] = Field(default="continue", description=(
        "continue: the model is told and carries on. fail: the task stops, "
        "checked at the next round boundary."))

    @property
    def gated(self) -> bool:
        """Whether the config alone gates this tool.

        Annotation rules can only add gates on top of this at boot, never remove
        one — so this is the floor, and what `charter validate` can report without
        talking to a server.
        """
        return self.approval == "always"


class ApprovalRules(Base):
    """What needs a human, by what the server says a tool does — rather than a
    per-tool entry for each of thirty tools.

    Resolved at boot from MCP's ToolAnnotations, and only ever able to *tighten*
    what the config decided. A server can make your agent safer without a deploy;
    making it less safe takes one. Same rule as Budget, which can narrow what policy
    allows and never widen it.
    """

    # read_only_hint: true
    read_only: Literal["never", "always"] = "never"
    # Everything else, including anything the server didn't annotate — which MCP
    # treats as destructive by default.
    default: Literal["never", "always"] = "always"


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
    # Set this and a tool without its own `approval` follows the server's
    # annotations instead of the `never` default.
    approval: ApprovalRules | None = None
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
        for t in self.tools:
            if not re.match(WIRE_TOOL_NAME, self.qualified(t.tool)):
                raise ValueError(
                    f"{self.qualified(t.tool)!r} is not a valid tool name — must "
                    f"match {WIRE_TOOL_NAME} once namespaced")
        return self

    def qualified(self, tool: str) -> str:
        """`server__tool`, not `server.tool`.

        Anthropic requires tool names to match ^[a-zA-Z0-9_-]{1,128}$ — a dot is
        rejected with a 400 before the model ever sees the request. Double
        underscore is the convention Claude Code uses for MCP tools (mcp__server__
        tool) for the same reason.

        One representation, used in the config, in tool_call_limits, in lifecycle
        rules, and on the wire. A prettier config name translated at the boundary
        would need converting in seven places, and missing one silently unenforces
        a limit.
        """
        return f"{self.name}{SEPARATOR}{tool}"

    @property
    def tool_names(self) -> list[str]:
        return [self.qualified(t.tool) for t in self.tools]


def split_qualified(qualified: str) -> tuple[str, str]:
    """`server__tool` -> ("server", "tool"). The inverse of McpServer.qualified,
    kept beside it so the two can't disagree about the separator."""
    server, _, tool = qualified.partition(SEPARATOR)
    return server, tool


class FieldSpec(Base):
    """One field of the agent's answer.

    Nests, because most answers aren't flat. A leads finder returns a *list of
    leads*, and flattening that into one formatted string turns a result you could
    query into prose someone has to re-parse. The schema goes straight to the
    harness as JSON Schema, which has always supported this — the restriction was
    ours, inherited from a time when the deliverable was a couple of scalars.
    """

    type: FieldType
    description: str | None = None
    items: "FieldSpec | None" = Field(default=None, description=(
        "What's in the list. Required when type is array."))
    properties: "dict[str, FieldSpec] | None" = Field(default=None, description=(
        "The object's fields. Required when type is object."))

    @model_validator(mode="after")
    def _check(self) -> FieldSpec:
        if self.type == "array" and self.items is None:
            raise ValueError("type: array needs `items` — an untyped list tells "
                             "the model nothing about what to put in it")
        if self.type == "object" and not self.properties:
            raise ValueError("type: object needs `properties`")
        if self.type not in ("array", "object") and (self.items or self.properties):
            raise ValueError(f"type: {self.type} takes neither items nor properties")
        return self


# The harness's own vocabulary, deliberately. deepagents groups its filesystem
# tools into `read` and `write`; `execute` and `spawn` are the two it ships
# without classifying, so we name those. Matching it exactly means a capability
# cap and a file rule can't drift into disagreeing about what `write` covers.
Capability = Literal["read", "write", "execute", "spawn"]

# A file rule only ever concerns the two the filesystem has.
FileOperation = Literal["read", "write"]


class FileRule(Base):
    """Which files the agent may touch, and whether a human is asked first.

    Versioned, not runtime policy: this changes what the agent can *reach*, so it
    has to roll back with the agent that reached it.

    The fields mirror the harness's own permission rules one-for-one — same
    operations, same glob semantics, same first-match-wins ordering — because the
    harness is what enforces them. Charter's contribution is that they're written
    down and versioned rather than living in whatever code built the agent.
    """

    operations: list[FileOperation] = Field(min_length=1)
    paths: list[str] = Field(min_length=1, description=(
        "Globs, absolute. Anchor the literal prefix (`/secrets/**`) — a fully "
        "unanchored pattern makes bulk tools like grep fire conservatively."))
    mode: Literal["allow", "deny", "interrupt"] = Field(default="allow", description=(
        "allow: proceeds. deny: the tool returns permission denied. interrupt: "
        "stops for human approval, durably."))

    @model_validator(mode="after")
    def _check(self) -> FileRule:
        for path in self.paths:
            if not path.startswith("/"):
                raise ValueError(f"file_rules: path {path!r} must start with '/'")
            if ".." in PurePosixPath(path).parts:
                raise ValueError(f"file_rules: path {path!r} must not contain '..'")
        return self


# Charter owns this wording rather than leaving each author to write a weaker
# version of it. A posture, deliberately, not a confidence threshold: a model's
# self-reported confidence isn't calibrated, so a number reads as precision that
# isn't there. Authorization is a decision a human makes in advance and writes
# down; confidence only decides whether the agent asks for help.
ASK_POSTURE = {
    "rarely": ("Ask only when you cannot proceed at all. Prefer making a "
               "reasonable choice and saying what you assumed."),
    "balanced": ("Ask when a wrong guess would be costly or hard to undo. "
                 "Otherwise decide, and say what you assumed."),
    "eagerly": ("When in doubt, ask. A question costs a few minutes; a wrong "
                "action can cost far more."),
}


class AskHuman(Base):
    """Let the agent stop and ask you something.

    The harness has no built-in way to do this — it only ever stops at a tool
    call — so asking *is* a tool, which is how Claude Code does it too. Declaring
    this adds one, gated so that calling it parks the run and hands back your
    answer.

    Off unless declared — omit the block and no ask tool exists, which is the only
    way to say "never" so there aren't two.
    """

    when: Literal["rarely", "balanced", "eagerly"] = Field(
        default="balanced", description=(
            "How readily it should interrupt you. A posture, not a threshold: a "
            "model's confidence in itself isn't calibrated enough to be a number."))
    timeout_seconds: int = Field(default=240, gt=0, description=(
        "How long you have before the agent is told nobody answered and carries "
        "on with what it has."))

    @property
    def guidance(self) -> str:
        return ASK_POSTURE.get(self.when, "")


# Everything the harness brings that a call can be stopped on. `submit_result` is
# injected by BoundFlow rather than deepagents, and it is the one worth naming: it
# is how an agent finishes, so gating it is how you approve the answer.
GATEABLE = frozenset({
    "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep",
    "execute", "task", "submit_result",
})


class Gate(Base):
    """How long a human has, and what happens if nobody answers.

    Charter no longer decides *when* to stop — the harness does, via a tool's
    `approval` or a `file_rules` interrupt. What Charter supplies is the waiting:
    the harness raises an in-process interrupt that dies with the worker, and we
    turn it into a durable gate that survives one.
    """

    timeout_seconds: int = Field(default=1800, gt=0, description=(
        "How long a human has before the action is refused on their behalf. "
        "A tool may override this with its own `approval_timeout_seconds`."))
    tools: list[str] = Field(default_factory=list, description=(
        "Harness tools that also stop for a human. `submit_result` gates the final "
        "answer itself — the approver sees the result, not a summary of it. "
        "Declared MCP tools use their own `approval:` field instead."))
    on_reject: Literal["continue", "fail"] = Field(default="continue", description=(
        "continue: the agent is told and carries on without that action. "
        "fail: the task stops. Choose fail when the gated action is the point — "
        "otherwise a task nobody approved still reports success."))

    @model_validator(mode="after")
    def _check_tools(self) -> Gate:
        # Validated because a typo here gates nothing, silently — the worst
        # possible failure for a field whose entire job is stopping something.
        if unknown := sorted(set(self.tools) - GATEABLE):
            raise ValueError(
                f"gate.tools: {', '.join(unknown)} — not a tool the harness "
                f"provides. One of: {', '.join(sorted(GATEABLE))}. Declared MCP "
                f"tools are gated with their own `approval: always`.")
        return self

    # One field rather than separate reject and timeout handling, because the
    # control plane gives us one branch: `AwaitApproval` has approve and reject and
    # nothing else, and an unanswered gate is resolved as a rejection. Nothing at
    # resume time can tell "nobody answered" from "someone said no" — both arrive
    # with no reason — so config shouldn't pretend to distinguish them.


class Schedule(Base):
    """Run this agent on a clock instead of waiting to be asked.

    Mutually exclusive with `inputs`: a periodic run has nobody to supply a ticket
    id, so an agent that needs one can't be scheduled. That agent looks for its own
    work instead — which is why a scheduled agent coalesces: two ticks that both
    mean "something changed, go look" really are one piece of work.
    """

    # "15m", "1h", "30s", "7d" — seconds are what BoundFlow takes, but nobody
    # thinks in seconds past about a minute.
    every: str = Field(description="30s, 15m, 1h, 7d.")
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


class AgentConfig(Base):
    apiVersion: Literal["charter/v1"]
    kind: Literal["AgentConfig"]

    name: str = Field(pattern=AGENT_NAME)
    version: int = Field(ge=1)
    description: str | None = None

    model: str = Field(min_length=1, description=(
        "Model id, e.g. claude-haiku-4-5. Needs pricing in worker.yaml or cost "
        "reads as zero, which silently disables every cost limit."))
    objective: str = Field(min_length=1, description=(
        "What this agent is responsible for, in plain English. Becomes the system "
        "prompt. Supports {{ inputs.<name> }} and nothing else."))

    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    schedule: Schedule | None = None
    mcp: list[McpServer] = Field(default_factory=list)
    # Structured output, handed to the harness as its response format. Omit and the
    # agent answers in prose, which is what most agents want.
    response_format: dict[str, FieldSpec] | None = None
    gate: Gate = Field(default_factory=Gate)
    ask_human: AskHuman | None = None

    # What the agent may do beyond the tools it declared. The harness ships its own
    # filesystem, so these bound it. Empty `allowed_capabilities` means no allowlist
    # rather than "nothing allowed" — a field that silently forbade everything the
    # moment someone added it would be a bad default.
    allowed_capabilities: list[Capability] = Field(default_factory=list, description=(
        "Default-deny allowlist over the harness's own tools. read | write | "
        "execute | spawn. Empty means unrestricted; declared MCP tools are always "
        "permitted."))
    file_rules: list[FileRule] = Field(default_factory=list)

    # Skills are not declared here. Anything under `v<N>/skills/` is shipped to the
    # agent's store and handed to the harness's own loader, so an author's existing
    # SKILL.md directories work unchanged and the config stays free of a manifest
    # that could drift from what's on disk. Versioning comes from `v<N>/` being
    # immutable, same as the yaml.

    @model_validator(mode="after")
    def _check(self) -> AgentConfig:
        self._check_servers_unique()
        self._check_templates()
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
        for where, text in sources:
            for ref in template_refs(text):
                if not ref.startswith("inputs."):
                    raise ValueError(
                        f"{where}: {{{{ {ref} }}}} — only {{{{ inputs.<name> }}}} is "
                        "supported")
                key = ref.split(".", 1)[1]
                if key not in self.inputs:
                    raise ValueError(f"{where}: {{{{ {ref} }}}} is not a declared input")

    @property
    def all_tools(self) -> list[str]:
        """Every declared tool, namespaced."""
        return [t for s in self.mcp for t in s.tool_names]

    @property
    def file_rules_interrupt(self) -> bool:
        """Whether any file rule stops for a human. Same question as `gated_tools`
        asks of MCP tools, and it needs the same notification route."""
        return any(r.mode == "interrupt" for r in self.file_rules)

    @property
    def gated_tools(self) -> list[str]:
        """Tools whose call stops for a human. They are still handed to the model —
        the harness interrupts at the call rather than hiding the capability."""
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
