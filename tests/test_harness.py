"""What Charter holds deepagents to, and where the harness could quietly stop.

These test the policy translation directly rather than through a whole task: the
harness is what enforces, so the thing worth pinning is that our rules reach its
own mechanisms — and that a rule doing its job doesn't look like a breakage.
"""

from __future__ import annotations

import pytest
from boundflow import AgentGovernor, RuntimePolicy, ToolCallLimit
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from charter.harness.callbacks import governed_tool_callbacks
from charter.harness.middleware import harness_call_limits

pytestmark = pytest.mark.asyncio


def calls_twice_then_stops() -> BaseChatModel:
    """A model that reaches for the same tool twice, then answers."""

    class CallsTwice(BaseChatModel):
        n: int = 0

        @property
        def _llm_type(self) -> str:
            return "fake"

        def bind_tools(self, *a, **k):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.n += 1
            if self.n <= 2:
                msg = AIMessage(content="", tool_calls=[{
                    "name": "ping", "id": f"c{self.n}", "type": "tool_call",
                    "args": {"x": "hi"}}])
            else:
                msg = AIMessage(content="done")
            return ChatResult(generations=[ChatGeneration(message=msg)])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
            return self._generate(messages, stop, run_manager, **kw)

    return CallsTwice()


async def test_a_spent_tool_cap_is_not_a_tool_failure():
    """A cap doing its job must not read as an integration breaking.

    It holds today for a reason that is easy to lose. A blocked call is synthesized
    in `after_model` and the tool never runs, so `BaseTool.run` never fires and the
    callbacks never see it — uncounted for free. A filesystem denial is different:
    the tool *does* run and refuses from inside its own body, which is why that one
    needs classifying.

    So the thing to watch isn't refusals generally, it's tools that police
    themselves internally. If a deepagents release moved cap enforcement into the
    tool body, spent caps would start counting as broken integrations — and since
    `tool_failure_counts` is what lifecycle rules read, an agent would be paused for
    having guardrails that work. The better they worked, the faster it would trip.
    """
    from deepagents import create_deep_agent
    from langchain_core.tools import tool as make_tool

    @make_tool
    async def ping(x: str) -> str:
        """Ping."""
        return "pong"

    gov = AgentGovernor(
        "capped",
        RuntimePolicy(tool_call_limits=[ToolCallLimit(tool="ping", max_calls=1)]),
        "m", collect_spans=False)

    agent = create_deep_agent(model=calls_twice_then_stops(), tools=[ping],
                              system_prompt="go",
                              middleware=harness_call_limits(gov))
    await agent.ainvoke({"messages": [HumanMessage(content="ping twice")]},
                        {"callbacks": [governed_tool_callbacks(gov)]})

    assert gov.calls_per_tool == {"ping": 1}, "the cap did not block the second call"
    assert gov.tool_failure_counts == {}, "a spent cap was counted as a failure"


async def test_a_tool_that_actually_breaks_is_counted():
    """The other half, and what stops the test above passing for the wrong reason.

    `tool_failure_counts == {}` would hold just as well if nothing were counting at
    all. This pins that the counting works, so an empty count there means a spent
    cap really wasn't recorded rather than that no failure ever could be.
    """
    from deepagents import create_deep_agent
    from langchain_core.tools import tool as make_tool

    @make_tool
    async def ping(x: str) -> str:
        """Ping, badly."""
        raise RuntimeError("the integration is down")

    gov = AgentGovernor("broken", RuntimePolicy(), "m", collect_spans=False)

    agent = create_deep_agent(model=calls_twice_then_stops(), tools=[ping],
                              system_prompt="go", middleware=[])
    # A tool that raises takes the whole invocation with it — the callback records
    # the failure on the way past, which is the part under test here.
    with pytest.raises(RuntimeError, match="integration is down"):
        await agent.ainvoke({"messages": [HumanMessage(content="ping twice")]},
                            {"callbacks": [governed_tool_callbacks(gov)]})

    assert gov.tool_failure_counts.get("ping"), "a broken tool went uncounted"
