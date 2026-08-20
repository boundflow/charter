"""Durable subagents: deepagents' async-subagent interface, over BoundFlow.

deepagents already defines what delegating to a long-running agent looks like —
`start_async_task`, `check_async_task`, `update_async_task`, `cancel_async_task`.
Its implementation talks to a LangGraph Platform server over the LangGraph SDK.
These are the same tools, pointed at a control plane instead: same names, same
arguments, same descriptions, same state channel, same semantics.

Almost none of the behaviour here is invented, and that is the point. The tool
descriptions are imported or copied verbatim, the argument schemas are subclassed
from theirs, the task record lives in their `async_tasks` channel behind their
reducer, and a task id the parent never started is refused because it simply isn't
in that channel — their ownership check, not one of ours. An agent written against
deepagents-on-Platform runs here unchanged.

What changes is what a child *is*. On Platform it is a thread on a server, and
whatever governs it is that deployment's business. Here it is a workflow: its own
budget, its own lifecycle policy, its own audit trail, and it survives the worker
that started it. So the same agent becomes governed by moving, which is the whole
distributed-harness claim in one interface.

Two things are genuinely ours. `spawns:` names which agents may be started, because
an agent that could create workflows freely would mint governed units nobody
budgeted for. And nothing reaps children when a parent ends — deliberate rather
than missing. A child outliving its parent is the point of starting one, it is what
deepagents does, and here it is already bounded: a child is an instance with its own
ceilings, so an orphan spends against those and stops. Cancelling one because its
parent finished first would destroy work that was still running correctly.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState

log = logging.getLogger(__name__)

#: Passed to a child so its runs land on the parent's thread. `durable_harness`
#: keys both the checkpoint and the store off `task_id`, so pinning it is what
#: makes a follow-up a continuation rather than a fresh start.
TASK_KEY = "task_id"


class Spawner:
    """Creates and follows durable children for one agent.

    Holds no state of its own: a task record is a `RequestInfo` the control plane
    already has, addressed by what the parent kept in `async_tasks`. A parent that
    resumes on another worker can still act on a child it started days ago.
    """

    def __init__(self, cp: Any, tenant_id: str, bundles: dict,
                 allowed: list[str]) -> None:
        self._cp = cp
        self._tenant_id = tenant_id
        self._bundles = bundles          # agent name -> AgentBundle
        self._allowed = list(allowed)

    @property
    def allowed(self) -> list[str]:
        """What this agent may spawn — listed in the start tool's description, the
        way deepagents lists its available types."""
        return list(self._allowed)

    # ── starting ────────────────────────────────────────────────────────────

    async def start(self, subagent_type: str, description: str,
                    tool_call_id: str) -> Any:
        """Create a child instance, set it going, and record it."""
        if subagent_type not in self._allowed:
            # Their wording for an unknown type, and like theirs a message rather
            # than an exception: the model picked a name it wasn't offered, which
            # it can recover from by picking another.
            return (f"Error: unknown subagent type '{subagent_type}'. "
                    f"Available types: {', '.join(self._allowed) or 'none'}")
        bundle = self._bundles.get(subagent_type)
        if bundle is None:
            # Declared but not served here — the operator's problem, not the
            # model's, and worded differently so nobody looks in the wrong config.
            return (f"'{subagent_type}' is declared in spawns but this worker "
                    f"doesn't serve it, so it can't be started.")

        from ..compile import compile_agent
        from ..provisioning.apply import create_instance

        workflow = await create_instance(
            self._cp, compile_agent(bundle), self._tenant_id)
        # The description is the child's work. It arrives as an input, so the child
        # references it the same way any other agent references one.
        task_id = await self._cp.invoke_workflow(
            workflow.id, context={"description": description})
        log.info("spawned %s: workflow=%s task=%s", subagent_type,
                 workflow.id[:8], task_id[:8])
        return _launched(f"Launched async subagent. task_id: {task_id}",
                         tool_call_id, {
                             "task_id": task_id,
                             "agent_name": subagent_type,
                             # Their field for "the handle you act on". Here that
                             # is the workflow, since the instance is the thing
                             # that can be re-invoked or deleted.
                             "thread_id": workflow.id,
                             # The run to ask about. A follow-up replaces it while
                             # task_id stays put, exactly as theirs does.
                             "run_id": task_id,
                             "status": "running",
                         })

    # ── following ───────────────────────────────────────────────────────────

    async def check(self, task_id: str, state: dict) -> str:
        """Status, and the result once there is one."""
        task = self._tracked(task_id, state)
        if isinstance(task, str):
            return task
        try:
            info = await self._cp.get_request_info(task["run_id"])
        except Exception as e:  # noqa: BLE001 — reported to the model, not raised
            return f"Failed to get run status: {e}"

        status = getattr(info.status, "value", info.status)
        if status not in ("completed", "failed"):
            return f"Task {task_id} is {status}. No result yet."
        return f"Task {task_id} finished ({status}). Result: {info.result}"

    async def update(self, task_id: str, message: str, state: dict,
                     tool_call_id: str) -> Any:
        """Send a follow-up, continuing the same conversation.

        A new run pinned to the same `task_id`, which is what `durable_harness`
        keys the checkpoint and the store from — so the child resumes on its own
        thread with its own filesystem and everything it said before, rather than
        starting over with extra instructions. The handle the parent holds doesn't
        change; only the run behind it does.
        """
        task = self._tracked(task_id, state)
        if isinstance(task, str):
            return task
        try:
            run_id = await self._cp.invoke_workflow(
                task["thread_id"],
                context={TASK_KEY: task_id, "description": message})
        except Exception as e:  # noqa: BLE001
            return f"Failed to send update: {e}"
        return _launched(f"Sent update to task {task_id}.", tool_call_id,
                         {**task, "run_id": run_id, "status": "running"})

    async def cancel(self, task_id: str, state: dict) -> str:
        task = self._tracked(task_id, state)
        if isinstance(task, str):
            return task
        try:
            await self._cp.delete_workflow(task["thread_id"])
        except Exception as e:  # noqa: BLE001
            return f"Failed to cancel run: {e}"
        return (f"Cancelled task {task_id}. If it was mid-run it finishes the "
                f"operation it was in first.")

    def _tracked(self, task_id: str, state: dict):
        """The task record, or a refusal if this parent never started it.

        deepagents' ownership check, and the reason it is worth having here rather
        than only there: these ids address workflows that spend real money, and a
        parent that could act on one it never started could cancel a sibling's work
        by guessing.
        """
        tasks = (state or {}).get("async_tasks") or {}
        task = tasks.get(task_id)
        if not task:
            known = ", ".join(sorted(tasks)) or "none"
            return (f"Error: no async task with id '{task_id}'. "
                    f"Tasks you started: {known}")
        return task


def _launched(message: str, tool_call_id: str, task: dict) -> Any:
    """A tool result that also writes the task into `async_tasks`.

    Their channel, their reducer, their field names — so it survives context
    compaction the way theirs does, and rides the checkpointer the harness is
    already given. That makes the record durable the moment the child exists,
    rather than whenever the operation happens to end.
    """
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    return Command(update={
        "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
        "async_tasks": {task["task_id"]: task},
    })


def spawn_middleware(spawner: Spawner | None) -> list:
    """Our tools, carrying deepagents' `async_tasks` state channel.

    A middleware rather than a bare tool list, because the channel comes with it:
    `state_schema` is how a tool gets to write somewhere the checkpointer will
    persist. Their class, so the reducer, the field names and the compaction
    behaviour are all theirs.
    """
    if spawner is None:
        return []
    from deepagents.middleware.async_subagents import AsyncSubAgentState
    from langchain.agents.middleware import AgentMiddleware

    class DurableSubagents(AgentMiddleware):
        state_schema = AsyncSubAgentState

        def __init__(self) -> None:
            super().__init__()
            self.tools = spawn_tools(spawner)

    return [DurableSubagents()]


def spawn_tools(spawner: Spawner | None) -> list:
    """deepagents' async-subagent tools, backed by the control plane.

    Descriptions and argument schemas come from deepagents rather than being
    restated: the schemas are subclassed to add the injected fields, which
    LangChain keeps out of what the model sees, so an agent written for theirs
    calls these with exactly the arguments it already knows.

    Injection has to go through the schema. `ToolRuntime` is detected on the
    function but stripped again when the call is validated against an explicit
    `args_schema`, so the runtime never arrives; a declared field survives
    validation and is filtered out of the model-facing schema.

    No `list_async_tasks`. Theirs reads the same channel we do, and there is no
    reason it couldn't exist — it just isn't needed yet, and an unused tool is one
    more thing in front of the model.
    """
    if spawner is None:
        return []
    from deepagents.middleware.async_subagents import (
        ASYNC_TASK_TOOL_DESCRIPTION,
        CancelAsyncTaskSchema,
        CheckAsyncTaskSchema,
        StartAsyncTaskSchema,
        UpdateAsyncTaskSchema,
    )
    from langchain_core.tools import StructuredTool

    class Start(StartAsyncTaskSchema):
        tool_call_id: Annotated[str, InjectedToolCallId]

    class Check(CheckAsyncTaskSchema):
        state: Annotated[dict, InjectedState]

    class Update(UpdateAsyncTaskSchema):
        state: Annotated[dict, InjectedState]
        tool_call_id: Annotated[str, InjectedToolCallId]

    class Cancel(CancelAsyncTaskSchema):
        state: Annotated[dict, InjectedState]

    # Plain functions rather than bound methods: injection is resolved from the
    # function's own annotations, and binding hides them.
    async def start(subagent_type: str, description: str, tool_call_id: str):
        return await spawner.start(subagent_type, description, tool_call_id)

    async def check(task_id: str, state: dict):
        return await spawner.check(task_id, state)

    async def update(task_id: str, message: str, state: dict, tool_call_id: str):
        return await spawner.update(task_id, message, state, tool_call_id)

    async def cancel(task_id: str, state: dict):
        return await spawner.cancel(task_id, state)

    def tool(name, coroutine, description, schema):
        return StructuredTool.from_function(
            coroutine=coroutine, name=name, description=description,
            args_schema=schema, infer_schema=False)

    available = "\n".join(f"- {a}" for a in spawner.allowed) or "- (none)"
    return [
        tool("start_async_task", start,
             ASYNC_TASK_TOOL_DESCRIPTION.format(available_agents=available),
             Start),
        # Copied verbatim from deepagents' own builders, which don't export them.
        tool("check_async_task", check,
             ("Check the status of an async subagent task. Returns the current "
              "status and, if complete, the result. Statuses shown earlier in the "
              "conversation are always stale, so call this to get the current "
              "status rather than reporting a status from a previous tool result."),
             Check),
        tool("update_async_task", update,
             ("Send updated instructions to an async subagent. Interrupts the "
              "current run and starts a new one on the same thread, so the "
              "subagent sees the full conversation history plus your new message. "
              "The task_id remains the same."),
             Update),
        # The one description that couldn't stay theirs. Theirs cancels the run
        # and keeps the thread, so the task can be picked up again; the control
        # plane has no run-level cancel, so ours disables the instance instead.
        # That is both less and more than theirs — work already executing runs to
        # the end of its operation, and the instance does not survive — and a model
        # told only "stop a task" would reasonably expect to check it afterwards.
        tool("cancel_async_task", cancel,
             ("Cancel an async subagent task you no longer need. This destroys the "
              "subagent and everything it remembered, so it cannot be checked or "
              "updated afterwards. Work already running finishes the step it is on "
              "rather than stopping instantly."),
             Cancel),
    ]
