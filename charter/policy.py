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
FILE_RULES = "file_rules"
ALLOWED_CAPABILITIES = "allowed_capabilities"
ALLOWED_TOOLS = "allowed_tools"


def build(cfg, per_run) -> dict[str, Any]:
    """Charter's policy vocabulary, as plain JSON-able data.

    Plain dicts rather than models: this crosses a wire that treats it as a struct,
    and a pydantic type here would only be re-parsed on the far side by us anyway.
    """
    custom: dict[str, Any] = {}

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
