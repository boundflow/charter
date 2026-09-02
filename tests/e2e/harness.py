"""A scripted chat model, for end-to-end tests that fake only the model.

`MockLlmClient` drove BoundFlow's own agent loop, which the harness replaced — it
speaks `LlmRequest`/`Turn`, and nothing sends those any more. What the harness
wants is a LangChain `BaseChatModel`, so that is what this is.

The model is the one component worth faking. Everything else in these tests is
real: a real control plane, a real MCP subprocess, real Postgres for the store and
checkpointer. Every bug this project has hit came from a boundary a fake didn't
model — dotted tool names, `is_error` under its 1.x name, an activate that the
fake accepted unconditionally — so the fakes stop at the model and no further.

    model = scripted(
        calls("desk__get_ticket", ticket_id="4821"),
        submits(resolution="refunded", refunded_usd=240),
    )

Each entry is one assistant turn. Once the script runs out the model answers in
prose, which ends any agent cleanly.

It deliberately does *not* repeat the last turn. Subagents share this model
instance, so a repeated tool call gets replayed by an agent that may not even have
that tool — which loops until the operation times out rather than failing in a way
anyone can read.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

SUBMIT_RESULT = "submit_result"


def calls(tool: str, **args) -> dict:
    """A turn that calls one tool."""
    return {"tool": tool, "args": args}


def submits(**fields) -> dict:
    """A turn that finishes, filling the injected result schema."""
    return {"tool": SUBMIT_RESULT, "args": fields}


def says(text: str) -> dict:
    """A turn with no tool call — the agent answering in prose."""
    return {"text": text}


def scripted(*turns: dict) -> BaseChatModel:
    """A chat model that plays `turns` in order and records what it was offered.

    `offered` is what makes the authority claim checkable from outside: it is the
    tool list actually put in front of the model, which is the thing a config file
    is ultimately promising something about.
    """

    class Scripted(BaseChatModel):
        n: int = 0
        offered: list[list[str]] = []
        received: list[Any] = []
        is_subagent: bool = False

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools: list, **kw: Any):
            # Recorded here rather than in _generate because binding is where the
            # harness decides what this agent may see.
            names = [getattr(t, "name", None) or t.get("name", "") for t in tools]
            self.offered.append(names)
            # A subagent shares this instance, so without telling them apart it
            # would eat turns written for the parent — and replay a tool call it
            # may not even have. Only the parent is given `task`.
            self.is_subagent = "task" not in names
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
            # Accumulated, not replaced. Subagents share this model instance, so a
            # subagent's own turn would otherwise overwrite the parent's history and
            # a test would assert against whatever happened to run last.
            self.received = list(self.received) + list(messages)
            if self.is_subagent:
                # Subagents answer and return, so the script belongs to the parent.
                turn = {"text": "subagent done"}
            else:
                # Which turn this is, read off the thread rather than counted on
                # this object. An operation is at-least-once — `resumable=True` — so
                # a re-dispatched round replays, and a counter would resume partway
                # through the script and run off the end, answering "done" where the
                # test expected a tool call.
                i = sum(1 for m in messages if isinstance(m, AIMessage))
                turn = turns[i] if i < len(turns) else {"text": "done"}
                self.n = i + 1
            # Usage is not optional. BoundFlow refuses to run a model that reports
            # none — it cannot price the call, and a cost cap that silently doesn't
            # apply is worse than no cap. So the fake reports it, which is the same
            # rule the fakes here follow everywhere: never be more permissive than
            # the real thing.
            usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
            if "text" in turn:
                msg = AIMessage(content=turn["text"], usage_metadata=usage)
            else:
                msg = AIMessage(content="", usage_metadata=usage, tool_calls=[{
                    "name": turn["tool"], "args": turn["args"],
                    "id": f"call_{self.n}", "type": "tool_call"}])
            return ChatResult(generations=[ChatGeneration(message=msg)])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
            return self._generate(messages, stop, run_manager, **kw)

    model = Scripted()
    model.offered = []
    model.received = []
    return model


def texts(messages) -> str:
    """Everything the model was told, flattened — content blocks included."""
    out = []
    for m in messages or []:
        c = getattr(m, "content", m)
        if isinstance(c, list):
            out += [str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in c]
        else:
            out.append(str(c))
    return "\n".join(out)


def factory(model: BaseChatModel):
    """`CharterWorker.chat_model` takes a factory, since the model name comes from
    each agent's versioned config rather than the worker manifest."""
    return lambda _name: model


def by_model(**scripts):
    """A factory that hands each agent its own script, keyed by model name.

    Needed once a parent and its child run at the same time: a single scripted
    model is a shared cursor, so the child consumes turns the parent was going to
    get and both go off the rails. Model name is the only thing a factory sees, so
    that's the key — give the agents in a test different models.
    """
    def build(name: str):
        if name not in scripts:
            raise AssertionError(
                f"no script for model {name!r} — this test's agents use "
                f"{', '.join(sorted(scripts))}")
        return scripts[name]
    return build
