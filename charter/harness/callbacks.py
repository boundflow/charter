"""Meter a harness's own tools through LangChain's callback system.

BoundFlow can only count what it dispatches, and a harness dispatches most of its own
tools — deepagents alone adds a filesystem, subagent spawning and skills. Without help
those calls are *inferred* from the model's request, which says a call happened but never
how it ended.

LangChain already has a surface built for exactly this: `on_tool_start` / `on_tool_end` /
`on_tool_error` fire inside `BaseTool.run`, around every tool a run executes. It is the
harness-native way to watch tool calls, and preferable to wrapping them in middleware for
one concrete reason — **it reaches subagents**. A parent agent's middleware doesn't wrap a
subagent's tool calls (deepagents compiles the subagent with its own middleware list), but
callbacks ride the runtime config down into it; deepagents says so itself, in
`middleware/subagents.py`:

    The parent's callbacks, tags and configurable reach the subagent automatically

For a product whose whole job is metering, a `task` call quietly spending an unmetered
budget is the failure that matters. So: callbacks observe, middleware enforces.

    agent.ainvoke(payload, {"configurable": {"thread_id": task_id},
                            "callbacks": [governed_tool_callbacks(governor)]})
"""
from __future__ import annotations

from typing import Any


# The harness reports a policy denial and a broken tool through the same field —
# `ToolMessage(status="error")` — so the only thing separating them is the text.
# Matching on it is unattractive and still the best option available: the failure
# mode is that a reword stops matching and denials go back to being counted, which
# is the behaviour without this. The alternative, re-evaluating the rules
# ourselves, can misclassify a *real* failure as a refusal — under-reporting, and
# a broken integration that looks healthy.
#
# Pinned to deepagents' wording: `f"Error: permission denied for {op} on {path}"`.
# `test_a_policy_denial_is_not_a_tool_failure` drives a real denial through the
# real middleware, so an upgrade that rewords this fails there rather than quietly
# reverting us.
_REFUSALS = ("error: permission denied",)


def _is_refusal(content) -> bool:
    """Whether an error result is policy saying no, rather than a tool breaking.

    They are different events and only one of them means anything is wrong.
    `tool_failure_counts` is what lifecycle rules read, so counting refusals there
    would pause an agent for having working guardrails — and the better they work,
    the faster it trips.
    """
    if isinstance(content, list):  # some providers return content blocks
        content = " ".join(str(b.get("text", "")) if isinstance(b, dict) else str(b)
                           for b in content)
    text = str(content or "").strip().lower()
    return any(text.startswith(r) for r in _REFUSALS)


def governed_tool_callbacks(governor):
    """A callback handler that records the harness's tool calls into the governor.

    Counts every tool the harness ran, and whether it failed, into the metrics that
    lifecycle rules read. Tools BoundFlow declared are skipped — their own wrapper
    already counted them.

    A factory rather than a class so importing `boundflow` doesn't require langchain.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class GovernedToolCallbacks(BaseCallbackHandler):
        # A spent failure budget raises `ToolFailureLimitExceeded`, and that has to
        # reach the caller rather than be swallowed as a broken callback.
        raise_error = True

        def __init__(self) -> None:
            # on_tool_end/on_tool_error carry the run, not the name; on_tool_start
            # carries the name. Bridge them by run_id.
            self._names: dict[Any, str] = {}

        def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
            self._names[run_id] = (serialized or {}).get("name", "")

        def on_tool_end(self, output, *, run_id, **kwargs) -> None:
            # A tool that caught its own failure returns an error `ToolMessage` rather
            # than raising, so success isn't implied by ending normally.
            failed = getattr(output, "status", None) == "error"
            if failed and _is_refusal(getattr(output, "content", "")):
                failed = False
            self._record(run_id, failed=failed)

        def on_tool_error(self, error, *, run_id, **kwargs) -> None:
            self._record(run_id, failed=True)

        def _record(self, run_id, *, failed: bool) -> None:
            name = self._names.pop(run_id, "")
            if not name or governor.is_governed_tool(name):
                return
            governor.record_harness_tool(name, failed=failed)

    return GovernedToolCallbacks()
