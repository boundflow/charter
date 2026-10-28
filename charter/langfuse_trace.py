"""Native LangChain tracing, including in-process subagents.

Use a dedicated provider so BoundFlow's optional operation exporter doesn't also
export these spans. Each resumed round has a real root in the same task trace.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar

_task_trace_id = ContextVar("charter_task_trace_id", default=None)


def tracer_provider():
    """Assign task IDs to real roots, without manufacturing an unexported parent."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

    class TaskIds(RandomIdGenerator):
        def generate_trace_id(self):
            return _task_trace_id.get() or super().generate_trace_id()

    return TracerProvider(id_generator=TaskIds())


def native_callback(public_key):
    """Keep the governance wrapper visible without charging for the model twice."""
    from langfuse.langchain import CallbackHandler

    class CharterCallback(CallbackHandler):
        def __init__(self):
            super().__init__(public_key=public_key)
            self.governed_models = set()
            self.open_tools = {}

        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
            self.open_tools[run_id] = parent_run_id
            return super().on_tool_start(serialized, input_str, run_id=run_id,
                                         parent_run_id=parent_run_id, **kwargs)

        def on_tool_end(self, output, *, run_id, **kwargs):
            try:
                return super().on_tool_end(output, run_id=run_id, **kwargs)
            finally:
                self.open_tools.pop(run_id, None)

        def on_tool_error(self, error, *, run_id, **kwargs):
            try:
                return super().on_tool_error(error, run_id=run_id, **kwargs)
            finally:
                self.open_tools.pop(run_id, None)

        def close_unfinished_tools(self, error):
            # Deep Agents cancellation can skip task's on_tool_error callback.
            # Close only observations whose actual starts we saw; never invent
            # calls, outputs or usage for the interrupted work.
            for run_id, parent in list(self.open_tools.items()):
                self.on_tool_error(error, run_id=run_id, parent_run_id=parent)

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            identity = (serialized or {}).get("id") or []
            if identity and identity[-1] == "GovernedChatModel":
                self.governed_models.add(run_id)
                return super().on_chain_start(serialized, {"messages": messages},
                                               run_id=run_id, **kwargs)
            return super().on_chat_model_start(serialized, messages, run_id=run_id, **kwargs)

        def on_llm_end(self, response, *, run_id, **kwargs):
            if run_id in self.governed_models:
                self.governed_models.remove(run_id)
                return super().on_chain_end({"generations": response.generations},
                                             run_id=run_id, **kwargs)
            return super().on_llm_end(response, run_id=run_id, **kwargs)

        def on_llm_error(self, error, *, run_id, **kwargs):
            if run_id in self.governed_models:
                self.governed_models.remove(run_id)
                return super().on_chain_error(error, run_id=run_id, **kwargs)
            return super().on_llm_error(error, run_id=run_id, **kwargs)

    return CharterCallback()


class LangfuseTracing:
    def __init__(self, spec, *, client=None, public_key=None):
        self.spec = spec
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        if client is None:
            missing = [name for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                                         "LANGFUSE_BASE_URL") if not os.getenv(name)]
            if missing:
                raise RuntimeError("Langfuse requires: " + ", ".join(missing))
            try:
                from langfuse import Langfuse
            except ImportError as exc:
                raise ImportError("Install 'boundflow-charter[langfuse]' for native tracing") from exc
            client = Langfuse(tracer_provider=tracer_provider(), sample_rate=1.0)
        self.client = client

    async def invoke(self, runnable, payload, config, *, ctx, agent, version, model=None):
        from langfuse import propagate_attributes
        from opentelemetry import trace

        task_id = str(ctx.context.get("task_id") or ctx.request_id)
        session = (ctx.context.get(self.spec.session_input)
                   if self.spec.session_input else None)
        session_id = str(session if session is not None else ctx.workflow_id)
        trace_id = hashlib.sha256(task_id.encode()).hexdigest()[:32]
        metadata = {"task_id": task_id, "workflow_id": str(ctx.workflow_id),
                    "agent": agent, "agent_version": str(version)}
        # Never reuse a callback between concurrent tasks: it tracks run IDs.
        callback = native_callback(self.public_key)
        config = {**config, "callbacks": [*(config.get("callbacks") or []), callback]}
        if model:
            config["metadata"] = {**(config.get("metadata") or {}), "ls_model_name": model}
        token = _task_trace_id.set(int(trace_id, 16))
        try:
            with trace.use_span(trace.INVALID_SPAN), propagate_attributes(
                session_id=session_id, metadata=metadata, tags=["charter", agent]
            ):
                with self.client.start_as_current_observation(
                    as_type="agent", name=f"{agent} v{version}", input=payload,
                ) as root:
                    try:
                        result = await runnable.ainvoke(payload, config)
                        root.update(output=result)
                        return result
                    except BaseException as error:
                        callback.close_unfinished_tools(error)
                        raise
        finally:
            _task_trace_id.reset(token)

    async def close(self):
        await asyncio.to_thread(self.client.shutdown)


@asynccontextmanager
async def tracing_lifecycle(spec):
    tracing = LangfuseTracing(spec) if spec is not None else None
    try:
        yield tracing
    finally:
        if tracing is not None:
            await tracing.close()
