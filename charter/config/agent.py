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
    on_reject: Literal["continue", "fail"] | None = Field(default=None, description=(
        "What a refusal of THIS tool means, overriding the agent's `gate.on_reject`. "
        "Rejecting an outreach message should let the rest of the run continue; "
        "rejecting a production deploy usually should not."))
    justify: bool = Field(default=True, description=(
        "Gated tools only. Adds a required `justification` argument the model must "
        "fill, which becomes the gate's justification and is stripped before the "
        "server is called. Turn it off for a call whose arguments already say "
        "everything, or where the extra tokens are not worth it."))

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


#: Hosts where plain HTTP is a deployment shape rather than an oversight. A
#: sidecar shares the pod's network namespace, so the traffic never leaves it.
_LOOPBACK = ("http://localhost", "http://127.0.0.1", "http://[::1]")


def _reachable(url: str) -> bool:
    if url.startswith("https://"):
        return True
    return any(url.startswith(f"{host}:") or url == host or
               url.startswith(f"{host}/") for host in _LOOPBACK)


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
    # Sent with every request to a `url` server. Values may reference the
    # environment as ${VAR} and usually should: a hosted server wants a bearer
    # token, and this file is committed.
    #
    #     headers:
    #       Authorization: "Bearer ${GITHUB_MCP_TOKEN}"
    headers: dict[str, str] = Field(default_factory=dict)
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
        if self.headers and not self.url:
            raise ValueError("`headers` is only valid alongside `url`")
        if self.url and not _reachable(self.url):
            raise ValueError(
                "`url` must be https://, or http:// on loopback — a sidecar in the "
                "same pod is reached over localhost, and anything else carrying a "
                "token in cleartext is a mistake")
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
    # `timeout_seconds` moved to runtime.yaml as `question_timeout_seconds`.

    @property
    def guidance(self) -> str:
        return ASK_POSTURE.get(self.when, "")


# Everything the harness brings that a call can be stopped on. `submit_result` is
# injected by BoundFlow rather than deepagents, and it is the one worth naming: it
# is how an agent finishes, so gating it is how you approve the answer.
GATEABLE = frozenset({
    "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep",
    "execute", "task", "submit_result",
    # Starting a durable child creates a governed unit with a budget of its own,
    # which the parent's ceiling does not bound — worth a signature in some shops.
    "start_async_task",
})


def duration_seconds(raw: str, where: str) -> int:
    """`30s`, `15m`, `1h`, `7d`. Seconds are what BoundFlow takes, but nobody
    thinks in seconds past about a minute."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    text = (raw or "").strip()
    if text[-1:] in units and text[:-1].isdigit():
        return int(text[:-1]) * units[text[-1]]
    raise ValueError(f"{where}: {raw!r} — use 30s, 15m, 1h, 7d")


class Wait(Base):
    """Let the agent stop and come back later.

    The harness has no way to do this, and structurally can't: waiting durably
    needs a scheduler, and it is a library. So it names the moment — the agent
    calls a tool — and BoundFlow does the waiting, which is the same split as
    approvals and questions.

    Nothing is running while it waits. The worker lets go, the task sleeps in
    Postgres, and it resumes on whatever worker is free with its whole
    conversation intact. Working time doesn't accrue, so a task that waits a week
    has spent none of its `max_seconds`.

    Off unless declared. The model chooses how long, because only it knows whether
    it's waiting on a person, a rate limit, or nothing.
    """
    # `max` moved to runtime.yaml as `max_wait_seconds`. Whether the agent can wait
    # is versioned — that changes its tool list — but how long it may sleep is an
    # operator's number, the same split `ask_human` already has. It never protected
    # anything the version promised, either: a longer sleep grants no capability, it
    # just holds a slot, which is exactly the kind of cost policy exists to tune.


class Subagent(Base):
    """A specialist this agent may delegate to, in-process.

    The harness always offers `general-purpose`: same tools as the parent, told
    what to do per call. A declared subagent is for when that is too much — a
    narrower tool list so it cannot wander, a cheaper model for grunt work, a
    standing prompt that says how to behave rather than repeating it in every
    description.

    Versioned rather than policy, because all three change what the agent *is*
    rather than bounding it. The bounds still come from policy: a subagent runs
    under the same `allowed_capabilities`, the same file rules, and the same
    capability allowances as its parent, and shares them rather than getting its
    own.

    It delegates in-process, so it cannot outlive the run or wait for a human.
    That is `start_async_task` and a separate agent.
    """

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{2,62}$")
    description: str = Field(min_length=1, description=(
        "What it is for. The parent reads this to decide when to delegate, so it "
        "should say when *not* to as well."))
    prompt: str | None = Field(default=None, description=(
        "Its standing instructions. Omit and it uses the harness's default."))
    tools: list[str] = Field(default_factory=list, description=(
        "Qualified names — `server__tool`. Must be tools this agent declares; a "
        "subagent cannot reach further than its parent. Empty means the parent's "
        "whole toolset, which is what general-purpose already gives you."))
    model: str | None = Field(default=None, description=(
        "Defaults to the agent's own. A cheaper one here is the usual reason to "
        "declare a subagent at all."))


class Gate(Base):
    """How long a human has, and what happens if nobody answers.

    Charter no longer decides *when* to stop — the harness does, via a tool's
    `approval` or a `file_rules` interrupt. What Charter supplies is the waiting:
    the harness raises an in-process interrupt that dies with the worker, and we
    turn it into a durable gate that survives one.
    """

    # `timeout_seconds` moved to runtime.yaml's `authority.approval_timeout_seconds`.
    # How long your approver has is a fact about your team, not about what the
    # agent does, and it should be tunable without cutting a version. A tool may
    # still override it with its own `approval_timeout_seconds`, which is the one
    # case where the window really is about the action.
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
        return duration_seconds(self.every, "every")

    @model_validator(mode="after")
    def _check(self) -> Schedule:
        self.every_seconds  # raises with the readable message
        return self


class SubagentFileOwnership(Base):
    write_tools: dict[str, list[str]] = Field(min_length=1,
        description='Declared write tool -> path argument names, including aliases. Only these tools are constrained.')
    max_files: int = Field(default=4, ge=1, le=100, strict=True)

    @model_validator(mode='after')
    def _check(self):
        if any(not fields or len(set(fields)) != len(fields) or
               any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', field) for field in fields)
               for fields in self.write_tools.values()):
            raise ValueError('write_tools needs nonempty unique path argument names')
        return self


class AgentConfig(Base):
    retain_tool_progress: bool = Field(default=False, strict=True, description=(
        'Retain bounded checkpointed invocation metadata for repeated_read_tools and declared '
        'ownership write tools. Show selectors/return outcomes after compaction without replacing '
        'source, blocking reads, granting permissions or claiming acceptance. Defaults disabled.'))
    normalize_tool_integer_arguments: bool = Field(default=False, strict=True, description=(
        'Opt in to canonical decimal-string transport for tool fields declared integer-only '
        'or nullable integer. Normal validation, bounds, required guards and permissions remain; '
        'actual conversions are logged and annotated on tool results. Other strings are unchanged.'))
    repeated_read_tools: list[str] = Field(default_factory=list, description=(
        'Opt-in read-only tools: three consecutive identical successful reads receive computed '
        'recovery feedback; six fail visibly. Changed arguments/results reset detection. '
        'No tool results are cached, replaced or removed.'))
    normalize_qwen_tool_fragments: bool = Field(default=False, strict=True, description=(
        'Opt in to strict schema-validated normalization of a whole Qwen native function '
        'fragment missing only its opening tool_call tag; ordinary tool governance applies.'))
    tool_format_retries: int = Field(default=0, ge=0, le=3, strict=True, description=(
        "Opt-in retries when the model prints known tool function tags without structured calls. "
        "Printed text is never executed; exhausted retries fail the task. Existing call budgets still apply."))
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
    initial_message: str = Field(
        default="Carry out the current task described above, using the supplied inputs and constraints.",
        min_length=1,
        description="First user-role message. Supports declared input templates; keep stable policy in objective.")

    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    schedule: Schedule | None = None
    mcp: list[McpServer] = Field(default_factory=list)
    tool_input_bindings: dict[str, dict[str, str]] = Field(default_factory=dict,
        description="Tool name -> argument -> declared input name. Caller-bound arguments are hidden from the model schema.")
    coordinator_tools: list[str] | None = Field(default=None,
        description="Optional declared-tool subset for the parent. Specialists retain their explicitly assigned tools.")
    coordinator_after_handoff_tools: list[str] = Field(default_factory=list,
        description='Optional coordinator tools deferred until its first specialist task returns. Specialists retain their tools; this routes initial diagnosis to them without changing acceptance requirements.')
    subagent_context_inputs: list[str] = Field(default_factory=list,
        description="Declared caller inputs automatically preserved in inline task descriptions. Opt in only to context appropriate for every specialist; never include credentials.")
    coordinator_context_inputs: list[str] = Field(default_factory=list,
        description="Declared caller inputs reattached to each coordinator model request after compaction. Opt in only to durable task constraints; never include credentials. This is not added to saved history.")
    subagent_required_tools: list[str] = Field(default_factory=list,
        description="Opt-in coordinator tool prerequisites for task delegation. Require successful declared tool receipts in retained history or checkpointed state, not prose claims. This does not validate result semantics.")
    subagent_review_tools: list[str] = Field(default_factory=list,
        description='Declared tools that by default must complete before first delegation, and always after each specialist batch. Tool success may report failed checks; this proves execution, not acceptance.')
    subagent_initial_review_required: bool = Field(default=True, strict=True,
        description='Require review before the first task (default). False permits explicitly requested implementation after required reads; fresh review after every specialist remains mandatory.')
    coordinator_final_review_retries: int = Field(default=0, ge=0, le=2, strict=True,
        description='Opt-in bounded model continuations when a coordinator attempts to finish without its required post-specialist review. Does not execute reviews, fabricate receipts, or permit unreviewed completion.')
    coordinator_assignment_input: str | None = Field(default=None,
        description='Optional declared boolean input enabling a narrow initial assignment view after an explicit failed-check receipt. Default disabled. Same model, task tool, budgets, ownership and fresh review remain.')
    coordinator_assignment_context_tools: list[str] = Field(default_factory=list,
        description='Declared read-only metadata tools whose actual bounded receipts accompany the assignment view. Full durable tool history remains retained.')
    coordinator_assignment_requires_failure: bool = Field(default=True, strict=True,
        description='Default: assignment requires explicit failed-check evidence. False routes user-requested implementation after required reads, without inferring any checker outcome; requires initial review to be deferred.')
    coordinator_assignment_contract_input: str | None = Field(default=None,
        description='Optional declared string input bound verbatim as the first assignment contract, replacing generated design notes. Original scope and permissions remain; never select credentials.')
    coordinator_assignment_diagnostic_tool: str | None = Field(default=None, description=(
        'Optional one initial independent review attempt after preparation and before first '
        'request-based assignment. Preserve its actual result or evidence gap; never replaces '
        'the mandatory fresh review after specialist work. Must be a declared review tool.'))
    coordinator_assignment_prepare_tool: str | None = Field(default=None,
        description='Optional declared coordinator tool offered alone after initial reads and before first assignment. Its actual setup receipt accompanies the handoff; preparation is not acceptance. Default disabled.')
    caller_deadline_input: str | None = Field(default=None,
        description='Optional declared string input containing an aware ISO caller deadline. A live reminder reaches coordinator and inline specialists; enforcement limits, tools and review requirements do not change. Default disabled.')
    subagent_required_read_tools: list[str] = Field(default_factory=list,
        description='Opt-in tools each inline specialist must complete in its own history or checkpointed state before using ownership-declared write tools. Does not validate relevance, freshness or understanding of evidence.')
    subagent_file_ownership: SubagentFileOwnership | None = Field(default=None,
        description='Opt-in owned_files task argument and exclusive write ownership for configured inline specialists. Not an OS sandbox.')
    builtin_tools_exclude: list[Literal['ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'execute']] = Field(
        default_factory=list, description="Built-in tools hidden and rejected for parent and inline specialists. Keep read_file if scratch history/tool-result retrieval is needed.")
    builtin_tool_hints: dict[Literal['ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'execute', 'task'], str] = Field(
        default_factory=dict, description="Optional routing hints prepended to built-in tool descriptions and appended to their error messages.")
    # Structured output, handed to the harness as its response format. Omit and the
    # agent answers in prose, which is what most agents want.
    response_format: dict[str, FieldSpec] | None = None
    wait: Wait | None = None
    subagents: list[Subagent] = Field(default_factory=list)
    gate: Gate = Field(default_factory=Gate)
    ask_human: AskHuman | None = None

    @model_validator(mode="before")
    @classmethod
    def _bare_blocks_are_on(cls, data):
        """`wait:` with nothing under it means on, not off.

        These two blocks say *whether* the agent gets a tool, and both now hold
        settings that are all optional — `wait` holds none at all. So the natural
        way to switch one on is a bare key, which YAML reads as null, which would
        otherwise mean absent, which means off. The agent would quietly lose a tool
        it plainly asked for, and the file would look right.

        Absent still means off. Only present-and-null is read as an empty block.
        """
        if not isinstance(data, dict):
            return data
        return {k: ({} if k in ("wait", "ask_human") and v is None else v)
                for k, v in data.items()}

    # `allowed_capabilities` and `file_rules` used to be declared here. They are
    # authority — what the agent may reach — and authority is policy: tightening it
    # should not require cutting a version, and it needs to be changeable on an
    # agent already running. They live in runtime.yaml's `authority:` block now.
    #
    # Keeping them here was worse than untidy. They were compiled into the runtime
    # policy *and* sealed into the version's artifact, the control plane's copy was
    # the one that bound, and the copy in the file did nothing — config that reads
    # as authoritative and is not.

    # `spawns` moved to runtime.yaml as `authority.allowed_spawns`. It is an
    # allowlist like `allowed_capabilities`, and deciding who may mint a governed
    # unit with a budget of its own should not have to wait for a release.

    # Skills are not declared here. Anything under `v<N>/skills/` is shipped to the
    # agent's store and handed to the harness's own loader, so an author's existing
    # SKILL.md directories work unchanged and the config stays free of a manifest
    # that could drift from what's on disk. Versioning comes from `v<N>/` being
    # immutable, same as the yaml.

    @model_validator(mode="after")
    def _check(self) -> AgentConfig:
        self._check_servers_unique()
        self._check_subagents()
        self._check_templates()
        if set(self.subagent_context_inputs) - set(self.inputs):
            raise ValueError('subagent_context_inputs must name declared inputs')
        if set(self.coordinator_context_inputs) - set(self.inputs):
            raise ValueError('coordinator_context_inputs must name declared inputs')
        if self.caller_deadline_input is not None:
            selected_deadline = self.inputs.get(self.caller_deadline_input)
            if selected_deadline is None or selected_deadline.type != 'string':
                raise ValueError('caller_deadline_input must name a declared string input')
        evidence_tools = set(self.subagent_required_tools) | set(self.subagent_review_tools)
        if self.coordinator_final_review_retries and (not self.subagent_review_tools or not self.subagents):
            raise ValueError('coordinator_final_review_retries requires review tools and configured specialists')
        if not self.subagent_initial_review_required and (not self.subagent_review_tools or not self.subagents):
            raise ValueError('subagent_initial_review_required=false requires review tools and configured specialists')
        if not self.coordinator_assignment_requires_failure:
            if self.coordinator_assignment_input is None or self.subagent_initial_review_required:
                raise ValueError('Request-based assignment requires coordinator_assignment_input and subagent_initial_review_required=false')
        assignment_tools = set(self.coordinator_assignment_context_tools)
        if self.coordinator_assignment_diagnostic_tool is not None:
            if (self.coordinator_assignment_input is None or self.coordinator_assignment_requires_failure
                    or not self.coordinator_assignment_prepare_tool
                    or self.coordinator_assignment_diagnostic_tool not in self.subagent_review_tools
                    or self.coordinator_assignment_diagnostic_tool == self.coordinator_assignment_prepare_tool):
                raise ValueError('coordinator_assignment_diagnostic_tool requires prepared request-based assignment and a distinct declared review tool')
            assignment_tools.add(self.coordinator_assignment_diagnostic_tool)
        if self.coordinator_assignment_prepare_tool is not None:
            if self.coordinator_assignment_input is None:
                raise ValueError('coordinator_assignment_prepare_tool requires coordinator_assignment_input')
            assignment_tools.add(self.coordinator_assignment_prepare_tool)
        if self.coordinator_assignment_contract_input is not None:
            contract = self.inputs.get(self.coordinator_assignment_contract_input)
            if self.coordinator_assignment_input is None or contract is None or contract.type != 'string':
                raise ValueError('coordinator_assignment_contract_input requires an assignment selector and a declared string input')
        if self.coordinator_assignment_input is not None:
            selected = self.inputs.get(self.coordinator_assignment_input)
            if selected is None or selected.type != 'boolean':
                raise ValueError('coordinator_assignment_input must name a declared boolean input')
            if not self.subagent_review_tools or not self.subagents:
                raise ValueError('coordinator_assignment_input requires review tools and configured specialists')
        elif assignment_tools:
            raise ValueError('coordinator_assignment_context_tools requires coordinator_assignment_input')
        if assignment_tools - set(self.all_tools):
            raise ValueError('coordinator_assignment_context_tools must name declared tools')
        if self.coordinator_tools is not None and assignment_tools - set(self.coordinator_tools):
            raise ValueError('coordinator_assignment_context_tools must be available to the coordinator')
        if self.subagent_file_ownership and assignment_tools & set(self.subagent_file_ownership.write_tools):
            raise ValueError('coordinator_assignment_context_tools cannot include ownership-declared write tools')
        deferred = set(self.coordinator_after_handoff_tools)
        if deferred - set(self.all_tools):
            raise ValueError('coordinator_after_handoff_tools must name declared tools')
        if deferred and not self.subagents:
            raise ValueError('coordinator_after_handoff_tools requires configured specialists')
        if deferred & evidence_tools:
            raise ValueError('coordinator_after_handoff_tools cannot defer delegation prerequisites')
        if self.coordinator_tools is not None and deferred - set(self.coordinator_tools):
            raise ValueError('coordinator_after_handoff_tools must be available to the coordinator')
        if evidence_tools - set(self.all_tools):
            raise ValueError('subagent_required_tools/subagent_review_tools must name declared tools')
        if self.subagent_file_ownership:
            if set(self.subagent_file_ownership.write_tools) - set(self.all_tools):
                raise ValueError('subagent_file_ownership must name declared tools')
            if not self.subagents:
                raise ValueError('subagent_file_ownership requires configured specialists')
        if self.coordinator_tools is not None and evidence_tools - set(self.coordinator_tools):
            raise ValueError('subagent_required_tools/subagent_review_tools must be available to the coordinator')
        if set(self.repeated_read_tools) - set(self.inline_tools):
            raise ValueError('repeated_read_tools must name declared inline tools')
        if self.subagent_file_ownership and set(self.repeated_read_tools) & set(self.subagent_file_ownership.write_tools):
            raise ValueError('repeated_read_tools must not include write tools')
        if self.subagent_required_read_tools:
            if not self.subagent_file_ownership:
                raise ValueError('subagent_required_read_tools requires subagent_file_ownership')
            reads = set(self.subagent_required_read_tools)
            if reads - set(self.inline_tools):
                raise ValueError('subagent_required_read_tools must name declared inline tools')
            if reads & set(self.subagent_file_ownership.write_tools):
                raise ValueError('subagent_required_read_tools must not include write tools')
            for specialist in self.subagents:
                inherited = self.coordinator_tools if self.coordinator_tools is not None else self.inline_tools
                if reads - set(specialist.tools or inherited):
                    raise ValueError('subagent_required_read_tools must be available to every specialist')
        if self.coordinator_tools is not None:
            unknown = set(self.coordinator_tools) - set(self.all_tools)
            if unknown:
                raise ValueError(f'coordinator_tools: unknown declared tools {sorted(unknown)}')
        for tool, bindings in self.tool_input_bindings.items():
            if tool not in self.all_tools:
                raise ValueError(f'tool_input_bindings: unknown declared tool {tool!r}')
            for argument, input_name in bindings.items():
                if not argument or input_name not in self.inputs:
                    raise ValueError(f'tool_input_bindings: {tool}.{argument} needs a declared input, got {input_name!r}')
        self._check_schedule()
        return self

    def _check_subagents(self) -> None:
        names = [s.name for s in self.subagents]
        if len(set(names)) != len(names):
            raise ValueError("two subagents share a name")
        if "general-purpose" in names:
            raise ValueError(
                "`general-purpose` is the harness's own subagent — declaring one "
                "by that name would replace it silently. Pick another name.")
        declared = set(self.all_tools)
        for sub in self.subagents:
            unknown = sorted(set(sub.tools) - declared)
            if unknown:
                raise ValueError(
                    f"subagent {sub.name!r} names tools this agent does not "
                    f"declare: {', '.join(unknown)}. A subagent cannot reach "
                    f"further than its parent.")

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
        sources = [("objective", self.objective), ("initial_message", self.initial_message)]
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
