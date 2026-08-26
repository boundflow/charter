"""Policy has to mean the same thing whoever is holding the tool.

deepagents compiles a subagent with its own middleware list, and
`SubAgentMiddleware` has no way to inherit the parent's. So a subagent Charter
does not configure runs outside the tool allowlist and the capability caps — and
an agent permitted to spawn could do through a child what it may not do itself.
"""

from __future__ import annotations

import pytest
from boundflow import AgentGovernor, RuntimePolicy

from charter.harness.durable import bounded_subagent
from charter.harness.middleware import harness_middleware


class Call:
    def __init__(self, tool: str) -> None:
        self.tool_call = {"name": tool, "id": "c1", "args": {}}


def offer(middleware, tool: str) -> str:
    """Run one tool call through a middleware stack."""
    for mw in middleware:
        out = mw.wrap_tool_call(Call(tool), lambda r: "RAN")
        if out != "RAN":
            return "refused"
    return "allowed"


def governor(**custom):
    return AgentGovernor("bounded", RuntimePolicy(custom=custom), "m",
                         collect_spans=False)


def test_a_subagent_carries_the_parents_allowlist():
    """The escape this exists to close: `allowed_capabilities: [read, spawn]` reads
    as safe, and without this the child could write."""
    gov = governor(allowed_capabilities=["read", "spawn"], allowed_tools=[])
    spec = bounded_subagent(gov, {})

    assert offer(spec["middleware"], "write_file") == "refused"
    assert offer(spec["middleware"], "read_file") == "allowed"


def test_a_subagent_is_deepagents_own_spec_plus_governance():
    """Spread rather than rebuilt, so the name, description, prompt and tool
    inheritance stay whatever deepagents says they are."""
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

    spec = bounded_subagent(governor(), {})

    for key, value in GENERAL_PURPOSE_SUBAGENT.items():
        assert spec[key] == value, f"{key} should be theirs, unchanged"
    assert "middleware" in spec and "permissions" in spec


def test_a_capability_allowance_is_the_agents_not_each_graphs():
    """A subagent gets its own middleware instances. A counter made inside one
    would be per-graph, so a parent and two children would each get the full
    allowance and a cap of two would admit six."""
    gov = governor(capability_call_limits=[{"capability": "write", "max_calls": 2}])
    spent: dict[str, int] = {}
    parent = harness_middleware(gov, spent)
    child = bounded_subagent(gov, spent)["middleware"]

    assert offer(parent, "write_file") == "allowed"
    assert offer(child, "write_file") == "allowed"
    assert offer(child, "write_file") == "refused", "the child spent the last one"
    assert offer(parent, "write_file") == "refused", "and the parent feels it"


def test_the_slot_is_claimed_before_the_call_runs():
    """Reading a total written *after* the call returns leaves a gap: two
    concurrent calls both see room and both proceed, so a cap of one admits as
    many as the model issued that turn. `begin_tool_call` claims per-tool slots up
    front for the same reason.
    """
    gov = governor(capability_call_limits=[{"capability": "write", "max_calls": 1}])
    spent: dict[str, int] = {}
    mw = harness_middleware(gov, spent)

    # Nothing has run — the handler is never invoked here — yet the second call is
    # already refused, which is only true if the first claimed its slot.
    assert offer(mw, "write_file") == "allowed"
    assert spent["write"] == 1
    assert offer(mw, "write_file") == "refused"


def test_a_failed_call_still_spends_its_slot():
    """Matching `begin_tool_call`, which does not decrement either: the allowance
    is on attempts, not successes. A tool that fails and is retried should not be
    free."""
    gov = governor(capability_call_limits=[{"capability": "write", "max_calls": 1}])
    spent: dict[str, int] = {}
    mw = harness_middleware(gov, spent)

    for middleware in mw:
        with pytest.raises(RuntimeError):
            middleware.wrap_tool_call(Call("write_file"),
                                      lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
        break

    assert spent["write"] == 1, "the failed attempt kept its slot"


# ── declared specialists ────────────────────────────────────────────────────


def agent_with(subagents):
    import yaml
    from charter.config.agent import AgentConfig
    from pathlib import Path
    raw = yaml.safe_load(
        (Path(__file__).parent.parent / "demo/leads/leads-finder/v1.yaml").read_text())
    raw["subagents"] = subagents
    return AgentConfig.model_validate(raw)


def fake_tools(*names):
    from langchain_core.tools import StructuredTool
    return [StructuredTool.from_function(func=lambda: "", name=n, description="t")
            for n in names]


def test_a_declared_subagent_carries_the_same_bounds():
    """The whole reason this exists rather than letting deepagents default it: a
    specialist that escaped the parent's policy would be a hole you opened on
    purpose."""
    from charter.harness.durable import declared_subagents

    cfg = agent_with([{"name": "researcher", "description": "Reads."}])
    gov = governor(allowed_capabilities=["read", "spawn"], allowed_tools=[])

    spec = declared_subagents(cfg, fake_tools("net__search_people"), gov, {})[0]

    assert offer(spec["middleware"], "write_file") == "refused"
    assert offer(spec["middleware"], "read_file") == "allowed"


def test_a_narrower_tool_list_is_honoured():
    from charter.harness.durable import declared_subagents

    cfg = agent_with([{"name": "researcher", "description": "Reads.",
                       "tools": ["net__search_people"]}])
    tools = fake_tools("net__search_people", "net__send_message")

    spec = declared_subagents(cfg, tools, governor(), {})[0]

    assert [t.name for t in spec["tools"]] == ["net__search_people"]


def test_declaring_no_tools_means_the_parents_whole_set():
    """Which is what general-purpose already is — so declaring one this way only
    makes sense alongside a different model or a standing prompt."""
    from charter.harness.durable import declared_subagents

    cfg = agent_with([{"name": "drafter", "description": "Writes.",
                       "model": "claude-haiku-4-5"}])

    spec = declared_subagents(cfg, fake_tools("a", "b"), governor(), {})[0]

    assert "tools" not in spec
    assert spec["model"] == "claude-haiku-4-5"


def test_a_subagent_cannot_reach_further_than_its_parent():
    import pytest as _pytest
    from pydantic import ValidationError

    with _pytest.raises(ValidationError, match="does not declare"):
        agent_with([{"name": "researcher", "description": "Reads.",
                     "tools": ["net__nonexistent"]}])


def test_general_purpose_cannot_be_redeclared():
    """It would replace the harness's own silently — including the one Charter
    supplies to carry the parent's policy."""
    import pytest as _pytest
    from pydantic import ValidationError

    with _pytest.raises(ValidationError, match="harness's own"):
        agent_with([{"name": "general-purpose", "description": "Mine now."}])


def test_every_subagent_shares_one_capability_allowance():
    """Per-graph counters would mean a cap of two admitting two per specialist."""
    from charter.harness.durable import bounded_subagent, declared_subagents

    cfg = agent_with([{"name": "researcher", "description": "Reads."}])
    gov = governor(capability_call_limits=[{"capability": "write", "max_calls": 1}])
    spent: dict[str, int] = {}

    general = bounded_subagent(gov, spent)["middleware"]
    specialist = declared_subagents(cfg, [], gov, spent)[0]["middleware"]

    assert offer(general, "write_file") == "allowed"
    assert offer(specialist, "write_file") == "refused"
