"""Against a real model. Everything else is real too.

The mock suite can't see the things a provider enforces. A dotted tool name —
`desk.get_ticket` — passed every unit test and every mock end-to-end test, then
400'd on the first live call because Anthropic requires ^[a-zA-Z0-9_-]{1,128}$.
That class of bug only surfaces here.

So these assert on *structure and acceptance*, not on wording: that the request we
build is one the provider accepts, that the injected output schema comes back
filled, that tokens are priced. What the model chooses to say is not under test.

    export ANTHROPIC_API_KEY=...
    pytest tests/e2e/test_live.py

Cheap on purpose — haiku, tiny budgets, two tests.
"""
from __future__ import annotations

import os

import pytest
from boundflow.anthropic_client import AnthropicLlmClient

from charter.provisioning.apply import apply_project
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_gate, wait_for_run
from tests.e2e.test_lifecycle import project  # noqa: F401

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"),
                       reason="ANTHROPIC_API_KEY not set"),
]


def recording_client():
    """A real client that remembers what tool names it was asked to send.

    Same trick the mock uses, and the reason it matters here: the charset failure
    happened in the request we built, so the request is the thing to look at.
    """
    client = AnthropicLlmClient(api_key=os.environ["ANTHROPIC_API_KEY"])
    inner = client.complete
    client.offered = []

    async def complete(request):
        client.offered.append([t.name for t in request.tools])
        return await inner(request)

    client.complete = complete
    return client


async def test_a_real_model_completes_a_task(cp, project):
    """The whole chain with nothing faked: tool names the provider accepts, the
    injected submit_result schema coming back filled, real tokens priced."""
    await apply_project(cp, project, only="ticket-sweeper")
    llm = recording_client()

    worker = CharterWorker(project, llm=llm)
    async with running(worker):
        wf = next(w for w in await cp.list_workflows()
                  if w.workflow_type == "ticket-sweeper")
        info = await wait_for_run(cp, await cp.invoke_workflow(wf.id), timeout=180)
    await worker.aclose()

    assert info.run_outcome.value == "successful", info.failure_reason
    # The deliverable is the schema we injected, filled by the model.
    assert isinstance(info.result["summary"], str) and info.result["summary"].strip()
    assert isinstance(info.result["needs_attention"], int)
    # Real tokens, priced with the tenant's pricing.
    assert info.result["cost_usd"] > 0

    # Every tool name we sent was one the provider accepted — no dots, no 400.
    import re
    from charter.config.agent import WIRE_TOOL_NAME
    assert llm.offered, "the model was never called"
    for names in llm.offered:
        for name in names:
            assert re.match(WIRE_TOOL_NAME, name), name


async def test_a_real_model_proposes_the_gated_tool_rather_than_calling_it(cp, project):
    """Ticket 4821 is two identical charges on one day, so a refund is the obvious
    move — and the only way the agent can take it is to ask.

    The assertion is that it *parks*, not what it says while parking. If the model
    decided no refund were warranted this would fail, which is a real (small) risk
    and the reason the fixture is unambiguous.
    """
    await apply_project(cp, project, only="refund-demo")
    llm = recording_client()

    worker = CharterWorker(project, llm=llm)
    async with running(worker):
        wf = next(w for w in await cp.list_workflows()
                  if w.workflow_type == "refund-demo")
        request_id = await cp.invoke_workflow(wf.id, context={"ticket_id": "4821"})

        gate = await wait_for_gate(cp, wf.id, timeout=180)
        assert "desk__create_refund" in gate.justification

        await cp.approve_workflow(wf.id, gate.approval_id, "live-test", "duplicate")
        info = await wait_for_run(cp, request_id, timeout=180)
    await worker.aclose()

    assert info.run_outcome.value == "successful", info.failure_reason
    assert info.result["acts_performed"][0]["tool"] == "desk__create_refund"

    # It could only ever propose it: the tool was never in a request we sent.
    for names in llm.offered:
        assert "desk__create_refund" not in names
        assert "desk__get_ticket" in names
