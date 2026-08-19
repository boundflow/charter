"""Bounding how many subagents an agent runs.

The harness has nothing purpose-built for this: `task` is just a tool to it, so a
count is `ToolCallLimitMiddleware` and a concurrency bound is `max_concurrency`,
which is a semaphore over *every* parallel graph task and would serialise ordinary
tool calls too. These bound `task` and nothing else.
"""

import asyncio

import pytest

from charter.config.runtime import PerRun
from charter.workflows.subagents import TASK_TOOL, subagent_limits


class Req:
    def __init__(self, name=TASK_TOOL):
        self.tool_call = {"name": name, "id": "c1"}


def limits(**kw):
    per_run = PerRun(max_cost_usd=1.0, **kw)
    made = subagent_limits(per_run)
    return made[0] if made else None


def run(coro):
    return asyncio.run(coro)


def test_no_limits_declared_adds_no_middleware():
    assert subagent_limits(PerRun(max_cost_usd=1.0)) == []


def test_a_total_is_cumulative_and_then_refuses():
    mw = limits(max_total_subagents=2)

    async def ok(_req):
        return "done"

    async def go():
        return [await mw.awrap_tool_call(Req(), ok) for _ in range(3)]

    first, second, third = run(go())
    assert first == "done" and second == "done"
    assert third.status == "error"
    assert "Subagent limit reached" in third.content


def test_other_tools_are_untouched():
    """Ordinary parallel tool calls are cheap and where the parallelism is a win —
    a subagent bound must not become a general throttle."""
    mw = limits(max_total_subagents=1)

    async def ok(_req):
        return "done"

    async def go():
        return [await mw.awrap_tool_call(Req("desk__get_ticket"), ok)
                for _ in range(5)]

    assert run(go()) == ["done"] * 5


def test_parallel_is_a_gauge_not_a_counter():
    """Fifty spawned in one turn are all in flight before any has recorded a cost,
    which no spend cap catches in time. But finished ones must free their slot."""
    mw = limits(max_parallel_subagents=2)
    peak = 0

    async def slow(_req):
        nonlocal peak
        peak = max(peak, mw.running)
        await asyncio.sleep(0.02)
        return "done"

    async def go():
        return await asyncio.gather(
            *(mw.awrap_tool_call(Req(), slow) for _ in range(5)))

    results = run(go())
    assert peak <= 2
    refused = [r for r in results if getattr(r, "status", None) == "error"]
    assert refused, "some should have been turned away"
    assert "running at once" in refused[0].content


def test_a_sequential_run_is_not_limited_by_the_parallel_bound():
    """One at a time never exceeds the gauge, however many there are."""
    mw = limits(max_parallel_subagents=1)

    async def ok(_req):
        return "done"

    async def go():
        return [await mw.awrap_tool_call(Req(), ok) for _ in range(6)]

    assert run(go()) == ["done"] * 6


def test_a_failed_subagent_frees_its_slot():
    """Otherwise a few failures wedge the gauge at the limit and the agent can
    never spawn again."""
    mw = limits(max_parallel_subagents=1)

    async def boom(_req):
        raise RuntimeError("subagent died")

    async def go():
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await mw.awrap_tool_call(Req(), boom)
        return mw.running

    assert run(go()) == 0
