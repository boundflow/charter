"""Charter's half of the runtime policy — the part BoundFlow carries but never reads.

`RuntimePolicy.custom` is opaque by contract: BoundFlow stores it, ships it with the
operation and hands it back, and enforces none of it. Everything in here is enforced
by Charter or by the harness it wires up, which is exactly why it can't be a typed
field over there — a control plane with a column for `allowed_capabilities` would be
one that knows what a deepagents capability is.

Writing and reading both live here so the two ends can't drift. `compile` builds the
dict; the harness's middleware and permissions read it back. A key spelled two ways
would be a policy that silently stops applying, and nothing would fail.
"""

from __future__ import annotations

from typing import Any

# Keys under `RuntimePolicy.custom`. Named once.
CAPABILITY_CALL_LIMITS = "capability_call_limits"
# Charter and the harness enforce these; BoundFlow has no field for them because
# they are this harness's vocabulary, not a control plane's. They travel so a
# worker holding only an artifact still has them — behaviour comes from the
# artifact, every limit comes from here.
MAX_SECONDS = "max_seconds"
MAX_TOOL_SECONDS = "max_tool_seconds"
MAX_TOTAL_SUBAGENTS = "max_total_subagents"
MAX_PARALLEL_SUBAGENTS = "max_parallel_subagents"
OPERATION_TIMEOUT_SECONDS = "operation_timeout_seconds"
FILE_RULES = "file_rules"
ALLOWED_CAPABILITIES = "allowed_capabilities"
ALLOWED_TOOLS = "allowed_tools"


def build(cfg, per_run, limits, operation_timeout: int) -> dict[str, Any]:
    """Charter's policy vocabulary, as plain JSON-able data.

    Plain dicts rather than models: this crosses a wire that treats it as a struct,
    and a pydantic type here would only be re-parsed on the far side by us anyway.
    """
    custom: dict[str, Any] = {}

    # Omitted when unset rather than sent as 0: `custom` is read with `.get`, and a
    # present zero would be indistinguishable from "no limit" at the reading end
    # for the four where 0 legitimately means unlimited.
    for key, value in (
        (MAX_SECONDS, per_run.max_seconds),
        (MAX_TOTAL_SUBAGENTS, per_run.max_total_subagents),
        (MAX_PARALLEL_SUBAGENTS, per_run.max_parallel_subagents),
    ):
        if value:
            custom[key] = value
    # These two always travel: both have a real default that is not "unlimited",
    # so leaving them out would quietly widen the agent's allowance.
    custom[MAX_TOOL_SECONDS] = limits.max_tool_seconds
    custom[OPERATION_TIMEOUT_SECONDS] = operation_timeout

    if per_run.capability_call_limits:
        custom[CAPABILITY_CALL_LIMITS] = [
            {"capability": l.capability, "max_calls": l.max_calls}
            for l in per_run.capability_call_limits]

    if cfg.file_rules:
        custom[FILE_RULES] = [
            {"operations": list(r.operations), "paths": list(r.paths), "mode": r.mode}
            for r in cfg.file_rules]

    if cfg.allowed_capabilities:
        custom[ALLOWED_CAPABILITIES] = list(cfg.allowed_capabilities)
        # Declared MCP tools are always permitted, so the allowlist only has to name
        # what the *harness* brings. Empty means no allowlist, not "nothing" — which
        # is why this is only set alongside a capability allowlist.
        custom[ALLOWED_TOOLS] = list(cfg.all_tools)

    return custom


def worker_limits(policy) -> dict[str, Any]:
    """The limits Charter itself enforces, as the loop wants them.

    Read from the policy rather than a local file so a worker serving a pulled
    artifact enforces the same numbers as one serving a checkout.
    """
    c = _of(policy)
    return {
        "max_seconds": c.get(MAX_SECONDS, 0),
        "max_tool_seconds": c.get(MAX_TOOL_SECONDS, 30.0),
        "max_total_subagents": c.get(MAX_TOTAL_SUBAGENTS, 0),
        "max_parallel_subagents": c.get(MAX_PARALLEL_SUBAGENTS, 0),
        "operation_timeout_seconds": c.get(OPERATION_TIMEOUT_SECONDS, 0),
    }


def _of(policy) -> dict[str, Any]:
    return getattr(policy, "custom", None) or {}


def file_rules(policy) -> list[dict]:
    return _of(policy).get(FILE_RULES) or []


def allowed_capabilities(policy) -> set[str]:
    return set(_of(policy).get(ALLOWED_CAPABILITIES) or [])


def allowed_tools(policy) -> set[str]:
    return set(_of(policy).get(ALLOWED_TOOLS) or [])


def capability_call_caps(policy) -> dict[str, int]:
    """Capability -> cap, for the middleware that enforces it.

    Replaces `governor.capability_call_caps()`, which went when the field did: a
    typed accessor for a custom key would put the SDK back in the business of
    knowing what a capability is.
    """
    return {l["capability"]: l["max_calls"]
            for l in _of(policy).get(CAPABILITY_CALL_LIMITS) or []
            if l.get("max_calls")}


def runtime_file(agent: str, policy) -> Any:
    """Rebuild `runtime.yaml`'s shape from the policy the control plane holds.

    A worker serving a pulled artifact has no runtime.yaml — that file is policy,
    it is applied rather than shipped, and putting it in the artifact would make a
    mutable thing look immutable. So the numbers come back the way they went out,
    and the loop reads the same object either way.
    """
    from .config.runtime import Limits, PerRun, RuntimePolicyFile

    typed = policy if not isinstance(policy, dict) else None
    def get(name, default=0):
        if typed is not None:
            return getattr(typed, name, default) or default
        return (policy or {}).get(name, default) or default

    mine = worker_limits(policy)
    failures = get("tool_failure_limits", []) or []
    per_tool = [f.max_failures if not isinstance(f, dict) else f["max_failures"]
                for f in failures]

    return RuntimePolicyFile(
        apiVersion="charter/v1", kind="RuntimePolicy", agent=agent,
        per_run=PerRun(
            max_cost_usd=get("max_cost_usd", 0.0),
            max_llm_calls=get("max_llm_calls", 0),
            max_seconds=mine["max_seconds"],
            max_total_subagents=mine["max_total_subagents"],
            max_parallel_subagents=mine["max_parallel_subagents"],
            # One number covers every tool, because that is how Charter declares it
            # — the policy carries it per tool only because BoundFlow's field is
            # shaped that way.
            max_tool_failures=max(per_tool) if per_tool else 0,
        ),
        limits=Limits(
            max_tokens_per_call=get("max_tokens_per_call", 1024),
            max_call_seconds=get("max_call_seconds", 60.0),
            max_tool_seconds=mine["max_tool_seconds"],
        ),
    )
