"""Which agent loop runs a round.

Charter owns the *outer* loop — rounds, gates, parks, durability. A harness owns
the *inner* one: call the model, dispatch tools, stop. That split is why swapping
harnesses is a one-line change here and invisible everywhere else.

Governance does not weaken when you swap. Since BoundFlow's `one enforcement core`
refactor, `run_step` and the governed path share a single `AgentGovernor` — caps,
per-tool limits, cost and spans have one implementation. BoundFlow's own loop is
just another client of it, which is what makes this general rather than bespoke.

Every harness here is a *prebuilt* agent taking a model and a list of tools. That's
the requirement for Charter: if a backend needed you to write a graph, picking it
wouldn't be a config choice, it would be code.
"""

from __future__ import annotations

from typing import Any, Literal

HarnessName = Literal["boundflow", "langgraph", "deepagents"]

# What each buys you, in the terms a config author would choose between.
DESCRIPTIONS: dict[str, str] = {
    "boundflow": (
        "BoundFlow's own loop. Straightforward: call the model, run the tools it "
        "asks for, stop when it submits. No planning, no sub-agents, no context "
        "compaction. The right default, and the only one with no extra dependency."),
    "langgraph": (
        "LangGraph's prebuilt ReAct agent. A standard reason/act loop with message "
        "state. Needs `langgraph` and a LangChain chat model."),
    "deepagents": (
        "LangChain's deep agent: planning, sub-agents, a virtual filesystem, and "
        "context management. Worth it for long, many-step tasks where a flat loop "
        "runs out of context. Needs `deepagents`."),
}


class HarnessUnavailable(RuntimeError):
    """The declared harness isn't installed, or its chat model isn't configured.

    Raised at boot rather than mid-task: an agent whose harness is missing is
    quarantined like one whose MCP server is missing, so it fails fast with a
    reason instead of discovering it a round in.
    """


def needs_chat_model(harness: str) -> bool:
    """Everything except BoundFlow's own loop drives a LangChain chat model."""
    return harness != "boundflow"


def check_available(harness: str) -> None:
    """Import-check a harness without building anything, so `charter worker` fails
    at startup with a fixable message."""
    if harness == "boundflow":
        return
    module, package = {
        "langgraph": ("langgraph.prebuilt", "langgraph"),
        "deepagents": ("deepagents", "deepagents"),
    }[harness]
    try:
        __import__(module)
    except ImportError as e:
        raise HarnessUnavailable(
            f"harness {harness!r} needs `pip install {package}`") from e


def build_invoke(harness: str, prompt: str, messages: list[dict]):
    """An `invoke(model, tools)` callable for `ctx.run_governed`.

    BoundFlow owns the call to this, so it can catch the injected `submit_result`
    finalizer and hand back a StepResult — meaning the happy path doesn't involve
    catching an exception, and the caps that force that finalizer are the same ones
    `run_agent` enforces.
    """
    if harness == "langgraph":
        from langgraph.prebuilt import create_react_agent

        async def invoke_langgraph(model: Any, tools: list) -> Any:
            agent = create_react_agent(model, tools, prompt=prompt)
            return await agent.ainvoke({"messages": messages})

        return invoke_langgraph

    if harness == "deepagents":
        from deepagents import create_deep_agent

        async def invoke_deepagents(model: Any, tools: list) -> Any:
            agent = create_deep_agent(model=model, tools=tools, instructions=prompt)
            return await agent.ainvoke({"messages": messages})

        return invoke_deepagents

    raise HarnessUnavailable(f"{harness!r} has no invoke adapter")
