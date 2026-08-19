"""Bound how many subagents an agent may run.

The harness gives you two general knobs and neither is about subagents.
`ToolCallLimitMiddleware` counts one tool cumulatively, which gets you a total but
nothing about how many are in flight. `max_concurrency` is a semaphore in the
graph executor, which bounds *every* parallel task — set it to hold subagents down
and you also serialise ordinary tool calls, which are cheap and where the
parallelism is a win.

Neither knows what a subagent is, because to the harness `task` is just a tool. So
this is ours: it wraps that one tool and nothing else.

Two numbers, stopping different things:

  max_total_subagents     a budget. An agent looping on subagents runs out of
                          money eventually, but stopping at a stated ceiling says
                          why, and says it before the money is gone.
  max_parallel_subagents  a valve. Fifty spawned in one turn are all in flight
                          before any of them has recorded a cost, so no spend cap
                          can catch that burst in time.

Over either limit the call is refused rather than raised — the model is told and
carries on with fewer helpers, which is how a spent cap behaves everywhere else.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# deepagents' name for the tool an agent calls to start a subagent.
TASK_TOOL = "task"


def subagent_limits(per_run) -> list:
    """Middleware enforcing the subagent bounds, or nothing if none are set."""
    if not (per_run.max_total_subagents or per_run.max_parallel_subagents):
        return []
    return [_subagent_middleware(per_run.max_total_subagents,
                                 per_run.max_parallel_subagents)]


def _subagent_middleware(total: int, parallel: int):
    from langchain.agents.middleware import AgentMiddleware

    class SubagentLimits(AgentMiddleware):
        """Counts spawns, and how many are running right now.

        State lives on the instance rather than in graph state because it only has
        to survive one operation: a task that parks and resumes rebuilds the graph
        anyway, and a fresh allowance per round is the more forgiving reading of a
        limit that exists to stop a burst.
        """

        def __init__(self) -> None:
            super().__init__()
            self.spawned = 0
            self.running = 0

        def _refusal(self, request):
            if total and self.spawned >= total:
                return (f"Subagent limit reached ({total} for this task). Do the "
                        f"remaining work yourself, or report what you have.")
            if parallel and self.running >= parallel:
                return (f"Too many subagents running at once (limit {parallel}). "
                        f"Wait for one to finish before starting another.")
            return None

        def _tool(self, request) -> str:
            call = getattr(request, "tool_call", None) or {}
            return call.get("name", "")

        async def awrap_tool_call(self, request, handler):
            if self._tool(request) != TASK_TOOL:
                return await handler(request)

            if (why := self._refusal(request)) is not None:
                log.info("subagent refused: %s", why)
                return _refuse(request, why)

            self.spawned += 1
            self.running += 1
            try:
                return await handler(request)
            finally:
                # In `finally` so a subagent that raised still frees its slot —
                # otherwise a few failures would wedge the gauge at the limit and
                # the agent could never spawn again.
                self.running -= 1

        def wrap_tool_call(self, request, handler):
            # The sync path exists for completeness; deepagents drives `task`
            # through the async one.
            if self._tool(request) != TASK_TOOL:
                return handler(request)
            if (why := self._refusal(request)) is not None:
                return _refuse(request, why)
            self.spawned += 1
            self.running += 1
            try:
                return handler(request)
            finally:
                self.running -= 1

    return SubagentLimits()


def _refuse(request, message: str):
    """A refusal the model can read, shaped like any other tool result."""
    from langchain_core.messages import ToolMessage

    call = getattr(request, "tool_call", None) or {}
    return ToolMessage(content=message, tool_call_id=call.get("id", ""),
                       name=call.get("name", TASK_TOOL), status="error")
