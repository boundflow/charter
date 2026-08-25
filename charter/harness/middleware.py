"""Bound which tools a harness may use at all.

Almost everything you'd want to do to a harness's tools, the harness already does better:

  * per-action allow / deny / interrupt      → deepagents `permissions=`
  * per-tool call counting and caps          → `ToolCallLimitMiddleware`
  * watching calls and how they ended        → LangChain callbacks, see
                                               `harness_callbacks.py`

Reimplementing any of those would put BoundFlow on the wrong side of the seam. So this
module mostly *translates*: it reads the agent's `RuntimePolicy` and hands the harness
its own mechanisms, pre-configured.

    agent = create_deep_agent(model=m, tools=t,
                              permissions=file_permissions(governor.policy),
                              middleware=harness_middleware(governor))

That is the value on offer, and it isn't enforcement — deepagents already enforces. It's
that the rules are *declarative and versioned*. The same constraints an engineer would
otherwise hard-code where the agent is constructed become policy that arrives with the
operation, changes when the agent version changes, and rolls back when the workflow rolls
back:

    max_llm_calls: 20
    capability_call_limits:
      - {capability: write, max_calls: 5}
    allowed_capabilities: [read, write]
    file_rules:
      - {operations: [write], paths: ["/secrets/**"], mode: deny}
      - {operations: [write], paths: ["/prod/**"], mode: interrupt}

One thing genuinely has no harness equivalent, and it's ours: a **default-deny
allowlist**. `permissions=` covers filesystem actions only, and `interrupt_on` can pause a
call but not refuse it, so "this agent may use exactly these tools and nothing else" has
nowhere else to live.

Ordering matters: LangChain composes "first in list as outermost layer", so these belong
first if they are to see everything.

One limit worth knowing before leaning on any of it: none of this reaches subagents.
deepagents compiles those with their own middleware list, so a `task` subagent is bounded
by its own spec, not this one — cap the `spawn` capability if that matters. Metering does
reach them, via callbacks.
"""
from __future__ import annotations

import logging

from .. import policy as charter_policy
from .capabilities import capability_of

log = logging.getLogger(__name__)


def tool_allowlist_middleware(governor):
    """Hold the harness to the policy's `allowed_tools` / `allowed_capabilities`.

        allowed_capabilities: [read]
        allowed_tools: [task]

    permits every filesystem read plus `task`, and refuses everything else the harness
    injected. Tools BoundFlow dispatches are always allowed; the customer named them by
    handing them over. Returns None when the policy sets no allowlist, so a caller can
    splice it in unconditionally.

    A factory rather than a class so importing `boundflow` doesn't require langchain —
    the dependency is only paid by callers who actually run a harness.
    """
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage

    tools = charter_policy.allowed_tools(governor.policy)
    caps = charter_policy.allowed_capabilities(governor.policy)
    if not tools and not caps:
        return None

    def refusal(request):
        """A `ToolMessage` refusing the call, or None to let it through.

        Refuse rather than raise: the model is told and adapts, which is what
        `run_agent` already does when a declared tool's cap is spent. A refused call
        never ran, so it is never metered.
        """
        name = _tool_name(request)
        if name in tools or capability_of(name) in caps or governor.is_governed_tool(name):
            return None
        log.debug("tool not in allowlist, refusing: tool=%s", name)
        return ToolMessage(
            content=(f"Tool '{name}' is not permitted for this agent. "
                     "Do not call it again."),
            tool_call_id=_call_id(request), status="error")

    class ToolAllowlistMiddleware(AgentMiddleware):
        """Holds the harness to a fixed set of tools."""

        name = "boundflow_tool_allowlist"

        async def awrap_tool_call(self, request, handler):
            return refusal(request) or await handler(request)

        def wrap_tool_call(self, request, handler):
            return refusal(request) or handler(request)

    return ToolAllowlistMiddleware()


def harness_middleware(governor, spent: dict[str, int] | None = None) -> list:
    """Every middleware the agent's policy calls for, outermost first.

    The one call a customer needs: whatever the policy happens to declare, this is what
    enforces it.
    """
    # One dict across the parent and every subagent it spawns, so a capability
    # allowance is the agent's rather than each graph's.
    spent = {} if spent is None else spent
    allowlist = tool_allowlist_middleware(governor)
    return ([allowlist] if allowlist else []) + harness_call_limits(governor, spent)


def harness_call_limits(governor, spent: dict[str, int] | None = None) -> list:
    """Turn the policy's call limits into middleware the harness runs.

    A cap naming a tool is handed to the harness's own `ToolCallLimitMiddleware`, using
    `run_limit` — a BoundFlow runtime policy bounds one agent run, and one graph
    invocation is one run. (A budget spanning every resume of a durable task is
    `thread_limit` instead; that is a different policy, not the one we have.)

    `capability_call_limits` can't be delegated, because `ToolCallLimitMiddleware` counts
    one tool or all tools and nothing in between, so that one is ours. It is also the one
    that survives the harness shipping a second way to do the same thing.

    Either way an over-cap call is refused and the agent keeps going, matching how a
    declared tool's spent cap behaves.
    """
    from langchain.agents.middleware import ToolCallLimitMiddleware

    spent = {} if spent is None else spent
    middleware = [ToolCallLimitMiddleware(tool_name=tool, run_limit=limit,
                                         exit_behavior="continue")
                  for tool, limit in governor.tool_call_caps().items()]
    by_capability = charter_policy.capability_call_caps(governor.policy)
    if by_capability:
        # Outermost, so a capability's budget is spent before any single tool's is.
        middleware.insert(0, _capability_limit_middleware(governor, by_capability, spent))
    return middleware


def _capability_limit_middleware(governor, limits: dict[str, int], spent: dict[str, int]):
    """Cap how many times an agent may do a *kind* of thing, however it does it.

    `spent` is counted here rather than read off `governor.calls_per_tool`, and it
    is passed in rather than made here, for two different reasons.

    Counted here, because reading a total that is written *after* the call returns
    leaves a gap: two concurrent calls both see `used < cap` and both proceed, so a
    cap of one admits as many calls as the model issued in that turn. Claiming the
    slot before the call runs closes it — the same thing `begin_tool_call` does for
    per-tool caps, and for the same reason. A call that then fails still counts,
    also matching: the allowance is on attempts, not successes.

    Passed in, because a subagent gets its own middleware instances. A counter made
    in here would be per-agent, so a parent and two children would each get the
    full allowance and the cap would mean three times what it says.
    """
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage

    def refusal(request):
        name = _tool_name(request)
        capability = capability_of(name)
        cap = limits.get(capability) if capability else None
        if cap is None:
            return None
        if spent.get(capability, 0) < cap:
            spent[capability] = spent.get(capability, 0) + 1
            return None
        log.debug("capability cap spent, refusing: tool=%s capability=%s used=%d cap=%d",
                  name, capability, spent.get(capability, 0), cap)
        return ToolMessage(
            content=(f"Limit reached: this agent may perform at most {cap} "
                     f"'{capability}' operations. '{name}' and every other "
                     f"{capability} tool are now unavailable."),
            tool_call_id=_call_id(request), status="error")

    class CapabilityLimitMiddleware(AgentMiddleware):
        name = "boundflow_capability_limits"

        async def awrap_tool_call(self, request, handler):
            return refusal(request) or await handler(request)

        def wrap_tool_call(self, request, handler):
            return refusal(request) or handler(request)

    return CapabilityLimitMiddleware()


def _tool_name(request) -> str:
    """The tool's name, however this LangChain version exposes it."""
    call = getattr(request, "tool_call", None) or getattr(request, "call", None) or {}
    if isinstance(call, dict) and call.get("name"):
        return call["name"]
    tool = getattr(request, "tool", None)
    return getattr(tool, "name", "") or ""


def _call_id(request) -> str:
    call = getattr(request, "tool_call", None) or getattr(request, "call", None) or {}
    return call.get("id", "") if isinstance(call, dict) else ""
