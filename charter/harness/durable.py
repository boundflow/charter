"""Wire a durable harness to an operation, without hand-rolling the keys.

A harness only survives the operation ending if two keys are right: the checkpointer's
`thread_id`, which continues the conversation, and the store namespace, which is the
agent's filesystem. Both are per *task*, and both are easy to get wrong in a way nothing
reports — reuse a namespace and two tasks quietly share files; vary a `thread_id` between
rounds and the agent starts over with no error anywhere.

So they aren't the caller's to choose. `durable_harness` derives both from the operation,
opens the Postgres store and checkpointer, and hands back everything a harness needs:

    async with durable_harness(ctx, "operator", STORE_URL) as h:
        result = await ctx.run_governed(
            "operator",
            lambda model, tools: create_deep_agent(
                model=model, tools=tools, system_prompt=SYSTEM, **h.wiring
            ).ainvoke(h.first({"messages": [...]}), h.config),
            chat_model=ChatAnthropic(model=MODEL),
            tools=[...])

`h.wiring` is backend, checkpointer, and the policy translated into the harness's own
mechanisms — permissions and middleware. `h.config` carries the thread and the metering
callbacks. `h.first(payload)` is the payload for a fresh round, or the resume command if
the task is parked, so the same call serves both.

Everything here is deepagents-shaped and imports langgraph directly — its conventions
used as they are, not wrapped in an abstraction over them.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from .callbacks import governed_tool_callbacks
from .capabilities import file_permissions
from .metering import metered
from .middleware import harness_middleware


@dataclass
class DurableHarness:
    """The wiring for one governed, durable round. Built by `durable_harness`."""

    thread_id: str
    wiring: dict
    config: dict
    _resume: Any = None
    _store: Any = None
    _saver: Any = None
    _namespace: tuple = ()

    async def discard(self) -> None:
        """Delete this task's state. Call it when the task is over, not before.

        Everything here exists to survive a gate, so it is dead weight the moment
        the task ends — and it is dead weight in the *operator's* database, which
        is the one place we can't see it accumulating to warn them. A daily agent
        would otherwise leave 365 conversations a year, each holding whatever it
        read.

        Deleting the thread takes the subagents with it: their checkpoints hang off
        the same `thread_id` under their own `checkpoint_ns`, so they'd otherwise
        be orphaned by a key nothing refers to any more.

        Never on the way to a gate. The state is the only reason a parked task can
        resume at all.
        """
        if self._saver is not None:
            await self._saver.adelete_thread(self.thread_id)
        if self._store is not None:
            for item in await self._store.asearch(self._namespace):
                await self._store.adelete(self._namespace, item.key)

    def first(self, payload: dict) -> Any:
        """The payload to invoke with: `payload` on a fresh task, or the parked
        interrupt's resume command when the operation is continuing one.

        Lets a handler open with the same line whether it's starting or resuming, which
        is the difference the caller most often forgets.
        """
        return self._resume if self._resume is not None else payload


def bounded_subagent(governor, spent: dict[str, int]) -> dict:
    """deepagents' general-purpose subagent, held to the same policy as its parent.

    Their spec, unchanged — name, standing prompt, and the parent's tools, which is
    what makes it general-purpose. Added: the same middleware and file permissions
    the parent runs under, so `allowed_capabilities` and the per-tool caps mean the
    same thing whoever is holding the tool.

    What the parent tells it to do is not fixed here. `task` takes a `description`
    per call — free text, "include all necessary context and specify the expected
    output format" — so the work is the parent's to choose every time. What a
    declared subagent would add is a different *standing* frame: its own persona, a
    narrower tool list, a cheaper model. That is a config surface Charter does not
    have yet, and it would attach governance the same way this does.
    """
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

    return {
        **GENERAL_PURPOSE_SUBAGENT,
        "middleware": harness_middleware(governor, spent),
        "permissions": file_permissions(governor.policy),
    }


@asynccontextmanager
async def durable_harness(ctx, agent_name: str, store_url: str, *, resume: Any = None):
    """Open the durable stores for this task and yield its wiring.

    `resume` is the decision from a gate — see `harness_gates` — and is what makes
    `h.first()` return a `Command` instead of a fresh message.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from langgraph.types import Command
    from deepagents.backends import StoreBackend

    governor = ctx.agent_governor(agent_name)
    governor.register_harness_observer()
    # Capability allowances are the agent's, not each graph's — so the parent and
    # every subagent it spawns count into the same dict.
    spent: dict[str, int] = {}

    # The task, not the operation: a resumed operation must land on the same thread and
    # the same filesystem as the one that parked.
    task_id = ctx.context.get("task_id") or ctx.request_id
    # The workflow *id*, not its type. Several workflows share a type — they're
    # instances of the same agent, each an entity with its own state — so keying on
    # type would interleave their namespaces under one prefix. Nothing collides
    # today, because task_id is unique, but deleting one instance could then only
    # be done by walking every task id and working out which belonged to whom.
    # Keyed on the id, an instance's state is a subtree you can drop.
    namespace = (ctx.workflow_id, agent_name, task_id)

    async with (
        AsyncPostgresStore.from_conn_string(store_url) as store,
        AsyncPostgresSaver.from_conn_string(store_url) as saver,
    ):
        await store.setup()
        await saver.setup()
        yield DurableHarness(
            thread_id=task_id,
            wiring={
                "backend": StoreBackend(namespace=lambda _rt: namespace, store=store),
                # Metered on the way through: the harness's own numbers are the
                # truth about spend, and they cover calls the governor never saw.
                "checkpointer": metered(saver, governor, ctx.report_metrics),
                # Policy, translated. Ours to declare and version, theirs to enforce.
                "permissions": file_permissions(governor.policy),
                "middleware": harness_middleware(governor, spent),
                # Supplied rather than defaulted, and only so the same bounds reach
                # it. deepagents adds a general-purpose subagent on its own, but a
                # subagent gets its own middleware list — `SubAgentMiddleware` has
                # no way to inherit the parent's — so a defaulted one runs outside
                # the tool allowlist and the per-tool caps. An agent allowed to
                # spawn could then do through a child what it may not do itself.
                "subagents": [bounded_subagent(governor, spent)],
            },
            config={
                "configurable": {"thread_id": task_id},
                # Metering rides callbacks so it reaches subagents, which a parent's
                # middleware never sees.
                "callbacks": [governed_tool_callbacks(governor)],
            },
            _resume=Command(resume=resume) if resume is not None else None,
            _store=store,
            _saver=saver,
            _namespace=namespace,
        )


class UngovernedModel(ValueError):
    """A subagent was configured with a model BoundFlow can't govern."""


def validate_subagents(specs) -> list:
    """Reject subagents that name their model as a string. Returns the specs.

        subagents=validate_subagents([RESEARCHER, SCRIBE])

    A spec inherits the parent's model *object* — the governed one — unless it names a
    model itself, in which case the harness builds its own client. That client never
    reaches the governor: its calls aren't capped, aren't priced, and don't exist in
    any metric until the checkpoint is read afterwards. Invisible money, and the only
    symptom is a cost limit that quietly doesn't apply.

    So it raises rather than warns. Omit `model` to inherit, or pass a model object if
    the subagent genuinely needs a different one — `ctx.agent_model()` returns a
    governed one.
    """
    for spec in specs:
        model = spec.get("model") if isinstance(spec, dict) else getattr(spec, "model", None)
        if isinstance(model, str):
            name = (spec.get("name") if isinstance(spec, dict) else None) or "<unnamed>"
            raise UngovernedModel(
                f"subagent {name!r} names its model as a string ({model!r}), which builds "
                "a client BoundFlow can't see: its calls are uncapped, unpriced and "
                "unmetered. Omit 'model' to inherit the governed one, or pass a model "
                "object.")
    return list(specs)


def task_context(ctx, extra: dict | None = None) -> dict:
    """Context for the next operation, carrying the task identity forward.

        return AwaitApproval(on_approve=Next("resume", context=task_context(ctx, {...})))

    Without this the resumed operation gets a new `request_id`, derives a different
    thread and namespace, and the agent silently starts from nothing.
    """
    return {"task_id": ctx.context.get("task_id") or ctx.request_id, **(extra or {})}
