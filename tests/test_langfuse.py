"""Real Deep Agents + real Langfuse SDK, exported locally without credentials."""
import asyncio
import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("langfuse")
from langfuse import Langfuse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool, ToolException
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from charter.config.worker import LangfuseConfig
from charter.langfuse_trace import LangfuseTracing, tracer_provider


class ScriptedBuilder(BaseChatModel):
    @property
    def _llm_type(self):
        return "scripted-builder"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        names = [getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "")
                 for t in kwargs.get("tools", [])]
        child = "task" not in names
        if any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content="Review complete" if child else "Build complete")
        elif child:
            msg = AIMessage(content="", tool_calls=[{
                "name": "probe", "id": str(uuid4()), "args": {"label": "helper"}}])
        else:
            msg = AIMessage(content="", tool_calls=[{
                "name": "task", "id": str(uuid4()),
                "args": {"description": f"Review component {n}",
                         "subagent_type": "general-purpose"}}
                for n in (1, 2)])
        msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    key = "pk-test-" + str(uuid4())
    client = Langfuse(public_key=key, secret_key="offline", base_url="http://localhost:1",
                      tracer_provider=tracer_provider(), span_exporter=exporter)
    tracing = LangfuseTracing(LangfuseConfig(session_input="project_id"),
                             client=client, public_key=key)
    yield tracing, exporter
    client.shutdown()


@pytest.mark.asyncio
async def test_two_helpers_tools_errors_usage_and_session_are_exported(tracing):
    from deepagents import create_deep_agent
    from boundflow import AgentGovernor, RuntimePolicy
    from charter.harness.durable import bounded_subagent
    from charter.harness.callbacks import governed_tool_callbacks
    from charter.config.runtime import PerRun
    from charter.workflows.subagents import subagent_limits

    active = peak = calls = 0

    @tool
    async def probe(label: str) -> str:
        """Check a component; the second check deliberately fails."""
        nonlocal active, peak, calls
        calls += 1
        index = calls
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.03)
        active -= 1
        if index == 2:
            raise ToolException("deliberate probe failure")
        return "component verified"

    probe.handle_tool_error = True
    gov = AgentGovernor("builder", RuntimePolicy(), "scripted")
    gov.register_harness_observer()
    from boundflow.langchain_client import GovernedChatModel
    model = GovernedChatModel(governor=gov, chat_model=ScriptedBuilder())
    agent = create_deep_agent(model=model, tools=[probe],
        subagents=[bounded_subagent(gov, {})],
        middleware=subagent_limits(PerRun(max_llm_calls=40, max_parallel_subagents=2, max_total_subagents=6)))
    observer = governed_tool_callbacks(gov)
    config = {"callbacks": [observer]}
    tracer, exporter = tracing
    ctx = SimpleNamespace(context={"project_id": "demo"}, request_id="run-one", workflow_id="wf")
    result = await tracer.invoke(agent, {"messages": [{"role": "user", "content": "Build"}]},
                                 config, ctx=ctx, agent="builder", version=1, model="test-model")
    tracer.client.flush()
    spans = exporter.get_finished_spans()
    assert result["messages"][-1].content == "Build complete"
    assert peak == 2 and calls == 2
    assert config["callbacks"] == [observer], "Do not mutate shared config"
    assert len({s.context.trace_id for s in spans}) == 1
    assert len([s for s in spans if s.name == "task"]) == 2
    assert len([s for s in spans if s.name == "probe"]) == 2
    generations = [s for s in spans if s.attributes.get("langfuse.observation.type") == "generation"]
    assert len(generations) == 6, [(s.name, dict(s.attributes)) for s in generations]
    assert any("deliberate probe failure" in str(s.attributes) for s in spans)
    assert any(s.name == "probe" and s.attributes.get("langfuse.observation.level") == "ERROR"
               for s in spans)
    assert any("component verified" in str(s.attributes) for s in spans)
    assert any("usage" in k for s in spans for k in s.attributes)
    assert any(s.attributes.get("gen_ai.request.model") == "test-model"
               or s.attributes.get("langfuse.observation.model.name") == "test-model"
               for s in spans)
    assert all(s.attributes.get("session.id") == "demo" for s in spans)
    ids = {s.context.span_id for s in spans}
    assert all(s.parent is None or s.parent.span_id in ids for s in spans)
    assert any(s.parent is None for s in spans), "Export a real root"
    by_id = {s.context.span_id: s for s in spans}
    for probe_span in [s for s in spans if s.name == "probe"]:
        ancestors = []
        current = probe_span
        while current.parent is not None:
            current = by_id[current.parent.span_id]
            ancestors.append(current.name)
        assert "task" in ancestors, "Helper tools must nest under the delegation"
    assert gov.calls_per_tool["probe"] == 2, "Keep Charter's callbacks"


@pytest.mark.asyncio
async def test_resumed_rounds_share_task_trace_but_projects_do_not_mix(tracing):
    from langchain_core.runnables import RunnableLambda
    tracer, exporter = tracing
    for task, request, project in [("task-a", "round-1", "a"),
                                   ("task-a", "round-2", "a"),
                                   ("task-b", "round-3", "b")]:
        ctx = SimpleNamespace(context={"task_id": task, "project_id": project},
                              request_id=request, workflow_id="wf")
        await tracer.invoke(RunnableLambda(lambda x: x), "hello", {}, ctx=ctx,
                            agent="builder", version=1)
    tracer.client.flush()
    roots = [s for s in exporter.get_finished_spans() if s.parent is None]
    assert len(roots) == 3
    assert roots[0].context.trace_id == roots[1].context.trace_id
    assert roots[0].context.trace_id != roots[2].context.trace_id
    assert roots[0].context.trace_id == int(hashlib.sha256(b"task-a").hexdigest()[:32], 16)


def test_enabled_tracing_requires_credentials(monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="LANGFUSE_PUBLIC_KEY"):
        LangfuseTracing(LangfuseConfig())


@pytest.mark.asyncio
async def test_cancelled_delegation_exports_its_parent_tool(tracing):
    from deepagents import create_deep_agent
    started = asyncio.Event()

    @tool
    async def probe(label: str) -> str:
        """Wait for the operator to interrupt this probe."""
        started.set()
        await asyncio.Event().wait()
        return 'unreachable'

    tracer, exporter = tracing
    agent = create_deep_agent(model=ScriptedBuilder(), tools=[probe])
    ctx = SimpleNamespace(context={'project_id': 'cancelled'}, request_id='cancelled-run', workflow_id='wf')
    task = asyncio.create_task(tracer.invoke(agent, {'messages': [{'role': 'user', 'content': 'Build'}]},
                                            {}, ctx=ctx, agent='builder', version=1))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel('operator suspension')
    with pytest.raises(asyncio.CancelledError):
        await task
    tracer.client.flush()
    spans = exporter.get_finished_spans()
    ids = {s.context.span_id for s in spans}
    assert all(s.parent is None or s.parent.span_id in ids for s in spans), [s.name for s in spans]
    assert len([s for s in spans if s.name == 'task']) == 2
