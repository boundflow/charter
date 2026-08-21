"""Against a real model. Everything else is real too.

The mock suite can't see what a provider enforces. A dotted tool name —
`desk.get_ticket` — passed every unit test and every mocked end-to-end test, then
400'd on the first live call because Anthropic requires ^[a-zA-Z0-9_-]{1,128}$.
That class of bug only surfaces here.

So these assert on *structure and acceptance*, not on wording: that the request we
build is one the provider accepts, that the injected output schema comes back
filled, and that a gated call still stops for a human when a real model is the one
reaching for it. What the model chooses to say is not under test.

    export ANTHROPIC_API_KEY=...
    pytest tests/e2e/test_live.py

Cheap on purpose — haiku, small budgets, two tests. They skip without a key, which
means a run with no key looks identical to a passing one; if these matter to you,
count the skips.
"""
from __future__ import annotations

import os
import re

import pytest
from langchain_anthropic import ChatAnthropic

from charter.config.agent import WIRE_TOOL_NAME
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_gate, wait_for_run
from tests.e2e.test_lifecycle import one_instance, project  # noqa: F401

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"),
                       reason="ANTHROPIC_API_KEY not set"),
]


def recording():
    """A real model factory that remembers the tool lists it was handed.

    `bind_tools` is where the harness decides what this agent may see, so it is the
    honest place to read the request from — and the charset failure was in the
    request we built, which is the thing worth looking at.

    A subclass rather than a wrapper: ChatAnthropic is a pydantic model and refuses
    attributes it has no field for.
    """
    offered: list[list[str]] = []

    class Recording(ChatAnthropic):
        def bind_tools(self, tools, **kw):
            offered.append([getattr(t, "name", None) or t.get("name", "")
                            for t in tools])
            return super().bind_tools(tools, **kw)

    def factory(model: str):
        return Recording(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])

    factory.offered = offered
    return factory


async def test_a_real_model_completes_a_task(cp, project, tenant):
    """The whole chain with nothing faked: tool names the provider accepts, and the
    injected submit_result schema coming back filled."""
    wf = await one_instance(cp, project, "ticket-sweeper", tenant)
    model = recording()

    worker = CharterWorker(project, chat_model=model)
    async with running(worker):
        info = await wait_for_run(cp, await cp.invoke_workflow(wf.id), timeout=180)
    await worker.aclose()

    assert info.run_outcome.value == "successful", info.failure_reason
    # The deliverable is the shape `response_format` declared, filled by the model.
    assert isinstance(info.result["summary"], str) and info.result["summary"].strip()
    # A number, not an int: protobuf Struct carries every number as a double, so a
    # field declared `type: integer` arrives back as a float. Asserting `== 2` would
    # hide that, which is how it went unnoticed — the mocked test does exactly that.
    count = info.result["needs_attention"]
    assert isinstance(count, (int, float)) and count == int(count)

    # Every name we sent was one the provider accepted — no dots, no 400.
    assert model.offered, "the model was never called"
    for names in model.offered:
        for name in names:
            assert re.match(WIRE_TOOL_NAME, name), name


async def test_a_real_model_proposes_the_gated_tool_rather_than_calling_it(
        cp, project, tenant):
    """Ticket 4821 is two identical charges on one day, so a refund is the obvious
    move — and the only way the agent can take it is to ask.

    The assertion is that it *parks*, not what it says while parking. If the model
    decided no refund were warranted this would fail, which is a real (small) risk
    and the reason the fixture is unambiguous.
    """
    wf = await one_instance(cp, project, "refund-demo", tenant)
    model = recording()

    worker = CharterWorker(project, chat_model=model)
    async with running(worker):
        request_id = await cp.invoke_workflow(wf.id, context={"ticket_id": "4821"})

        gate = await wait_for_gate(cp, wf.id, timeout=180)
        assert "desk__create_refund" in gate.justification

        await cp.approve_workflow(wf.id, gate.approval_id, "live-test", "duplicate")
        info = await wait_for_run(cp, request_id, timeout=180)
    await worker.aclose()

    assert info.run_outcome.value == "successful", info.failure_reason
    assert isinstance(info.result["refunded_usd"], (int, float))

    # It was offered, and still couldn't be called without a human. Omission was
    # the old mechanism; under the harness the tool is in the list and the *call*
    # is what stops, so a test that asserted absence would now pass for the wrong
    # reason.
    assert any("desk__create_refund" in names for names in model.offered)
