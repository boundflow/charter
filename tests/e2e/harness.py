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

Each entry is one assistant turn. The last one repeats if the agent keeps going,
so a script never runs out mid-test — a test that wanted a specific ending says
so by ending on `submits`.
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

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools: list, **kw: Any):
            # Recorded here rather than in _generate because binding is where the
            # harness decides what this agent may see.
            self.offered.append([getattr(t, "name", None) or t.get("name", "")
                                 for t in tools])
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
            turn = turns[min(self.n, len(turns) - 1)]
            self.n += 1
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
    return model


def factory(model: BaseChatModel):
    """`CharterWorker.chat_model` takes a factory, since the model name comes from
    each agent's versioned config rather than the worker manifest."""
    return lambda _name: model
