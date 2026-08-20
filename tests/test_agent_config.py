from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from charter.config.agent import AgentConfig

EXAMPLE = Path(__file__).parent.parent / "examples" / "refund-triage" / "v1.yaml"


def load(**overrides) -> dict:
    """The example config as a dict, with `overrides` merged one level deep."""
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw.update(overrides)
    return raw


def test_example_parses():
    cfg = AgentConfig.model_validate(load())
    assert cfg.name == "refund-triage"
    assert cfg.version == 1


def test_derived_views():
    cfg = AgentConfig.model_validate(load())
    assert cfg.gated_tools == ["stripe__create_refund"]
    assert "zendesk__get_ticket" in cfg.inline_tools
    assert "stripe__create_refund" not in cfg.inline_tools
    assert cfg.fail_fast_tools == {"zendesk__get_ticket", "stripe__create_refund"}
    assert len(cfg.all_tools) == 7


def test_invoke_mode_is_derived():
    cfg = AgentConfig.model_validate(load())
    assert cfg.invoke_mode == "queue"  # declares inputs -> discrete tasks

    raw = load()
    raw["inputs"] = {}
    raw["objective"] = "Look for anything that needs attention."
    assert AgentConfig.model_validate(raw).invoke_mode == "coalesce"


def test_unknown_key_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        AgentConfig.model_validate(load(retries=3))


class TestTemplating:
    def test_undeclared_input_rejected(self):
        raw = load(objective="Resolve {{ inputs.nonexistent }}.")
        with pytest.raises(ValidationError, match="not a declared input"):
            AgentConfig.model_validate(raw)

    def test_non_input_namespace_rejected(self):
        """The old {{ propose.* }} namespace pointed at runtime data you couldn't
        find by reading the file. Only inputs are referenceable now."""
        raw = load(objective="Approve {{ propose.tool }}?")
        with pytest.raises(ValidationError, match="only .* inputs"):
            AgentConfig.model_validate(raw)

class TestMcpServer:
    def test_command_and_url_together_rejected(self):
        raw = load()
        raw["mcp"][0]["url"] = "https://example.com"
        with pytest.raises(ValidationError, match="exactly one of"):
            AgentConfig.model_validate(raw)

    def test_neither_command_nor_url_rejected(self):
        raw = load()
        raw["mcp"][0].pop("command")
        raw["mcp"][0].pop("args")
        with pytest.raises(ValidationError, match="exactly one of"):
            AgentConfig.model_validate(raw)

    def test_http_url_rejected(self):
        raw = load()
        raw["mcp"][1]["url"] = "http://mcp.stripe__com"
        with pytest.raises(ValidationError, match="https"):
            AgentConfig.model_validate(raw)

    def test_env_value_rejected(self):
        """`env` takes variable NAMES — this file is committed and immutable, so a
        literal secret here would live forever."""
        raw = load()
        raw["mcp"][1]["env"] = ["sk_live_abc123"]
        with pytest.raises(ValidationError, match="variable NAME"):
            AgentConfig.model_validate(raw)

    def test_duplicate_tool_rejected(self):
        raw = load()
        raw["mcp"][0]["tools"].append({"tool": "get_ticket"})
        with pytest.raises(ValidationError, match="duplicate tool"):
            AgentConfig.model_validate(raw)


class TestInputSpec:
    def test_required_with_default_rejected(self):
        raw = load()
        raw["inputs"]["ticket_id"]["default"] = "123"
        with pytest.raises(ValidationError, match="mutually exclusive"):
            AgentConfig.model_validate(raw)

    def test_default_type_mismatch_rejected(self):
        raw = load()
        raw["inputs"]["max_refund_usd"]["default"] = "lots"
        with pytest.raises(ValidationError, match="is not a number"):
            AgentConfig.model_validate(raw)

    def test_default_outside_enum_rejected(self):
        raw = load()
        raw["inputs"]["priority"]["default"] = "critical"
        with pytest.raises(ValidationError, match="not in `enum`"):
            AgentConfig.model_validate(raw)

    def test_bool_is_not_an_integer(self):
        raw = load()
        raw["inputs"]["max_refund_usd"]["default"] = True
        with pytest.raises(ValidationError, match="is not a number"):
            AgentConfig.model_validate(raw)


class TestSchedule:
    def test_periodic_agent(self):
        raw = load()
        raw.pop("inputs")
        raw["objective"] = "Look for anything that needs attention."
        raw["schedule"] = {"every": "15m"}
        cfg = AgentConfig.model_validate(raw)
        assert cfg.schedule.every_seconds == 900
        assert cfg.schedule.manual is True  # still testable by hand

    @pytest.mark.parametrize("spec,seconds",
                             [("30s", 30), ("15m", 900), ("1h", 3600), ("7d", 604800)])
    def test_durations(self, spec, seconds):
        raw = load()
        raw.pop("inputs")
        raw["objective"] = "Look."
        raw["schedule"] = {"every": spec}
        assert AgentConfig.model_validate(raw).schedule.every_seconds == seconds

    def test_a_bad_duration_says_what_to_write(self):
        raw = load()
        raw.pop("inputs")
        raw["objective"] = "Look."
        raw["schedule"] = {"every": "every 15 minutes"}
        with pytest.raises(ValidationError, match="use 30s, 15m, 1h, 7d"):
            AgentConfig.model_validate(raw)

    def test_schedule_and_inputs_are_mutually_exclusive(self):
        """A periodic run has nobody to supply a ticket id."""
        raw = load(schedule={"every": "15m"})
        with pytest.raises(ValidationError, match="mutually exclusive"):
            AgentConfig.model_validate(raw)


class TestWireToolNames:
    def test_a_dot_would_be_rejected_by_the_provider(self):
        """Anthropic requires ^[a-zA-Z0-9_-]{1,128}$. A dotted namespace 400s
        before the model sees the request, so the config can't produce one."""
        import re
        from charter.config.agent import WIRE_TOOL_NAME
        cfg = AgentConfig.model_validate(load())
        for tool in cfg.all_tools:
            assert re.match(WIRE_TOOL_NAME, tool), tool
        assert not re.match(WIRE_TOOL_NAME, "stripe.create_refund")

    def test_a_server_name_that_would_break_the_wire_name_is_rejected(self):
        raw = load()
        raw["mcp"][0]["name"] = "a" * 130
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(raw)


class TestFileRules:
    """The harness ships a filesystem, so these bound it. Versioned rather than
    runtime policy, because they change what the agent can reach."""

    def test_paths_must_be_absolute(self):
        with pytest.raises(ValidationError, match="must start with"):
            AgentConfig.model_validate(load(file_rules=[
                {"operations": ["write"], "paths": ["secrets/**"], "mode": "deny"}]))

    def test_traversal_rejected(self):
        with pytest.raises(ValidationError, match=r"must not contain"):
            AgentConfig.model_validate(load(file_rules=[
                {"operations": ["write"], "paths": ["/data/../etc"], "mode": "deny"}]))

    def test_an_interrupt_rule_needs_an_approval_route(self):
        """Same question `gated_tools` asks, so it wants the same warning."""
        cfg = AgentConfig.model_validate(load(file_rules=[
            {"operations": ["write"], "paths": ["/prod/**"], "mode": "interrupt"}]))
        assert cfg.file_rules_interrupt

    def test_allow_is_the_default(self):
        cfg = AgentConfig.model_validate(load(file_rules=[
            {"operations": ["read"], "paths": ["/data/**"]}]))
        assert cfg.file_rules[0].mode == "allow"
        assert not cfg.file_rules_interrupt


class TestCapabilities:
    def test_allowlist_is_optional(self):
        """Empty means no allowlist, not "nothing allowed" — a field that silently
        forbade everything the moment someone added it would be a bad default."""
        cfg = AgentConfig.model_validate(load(allowed_capabilities=[]))
        assert cfg.allowed_capabilities == []

    def test_vocabulary_is_closed(self):
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(load(allowed_capabilities=["network"]))

    def test_matches_the_harness_vocabulary(self):
        """Ours must equal deepagents' filesystem operations plus the two it ships
        without classifying, or a cap and a file rule could disagree about `write`."""
        cfg = AgentConfig.model_validate(
            load(allowed_capabilities=["read", "write", "execute", "spawn"]))
        assert cfg.allowed_capabilities == ["read", "write", "execute", "spawn"]


class TestResponseFormat:
    """An answer isn't always flat. A leads finder returns a list of leads, and
    flattening that into one string turns a result you could query into prose
    someone has to re-parse."""

    def test_a_list_of_objects(self):
        cfg = AgentConfig.model_validate(load(response_format={
            "leads": {"type": "array", "items": {"type": "object", "properties": {
                "handle": {"type": "string"}, "pain": {"type": "string"}}}}}))
        leads = cfg.response_format["leads"]
        assert leads.items.properties["handle"].type == "string"

    def test_an_untyped_list_is_refused(self):
        """`type: array` alone tells the model nothing about what to put in it."""
        with pytest.raises(ValidationError, match="needs `items`"):
            AgentConfig.model_validate(load(response_format={
                "leads": {"type": "array"}}))

    def test_an_object_needs_its_fields(self):
        with pytest.raises(ValidationError, match="needs `properties`"):
            AgentConfig.model_validate(load(response_format={
                "lead": {"type": "object"}}))

    def test_a_scalar_takes_neither(self):
        with pytest.raises(ValidationError, match="neither items nor properties"):
            AgentConfig.model_validate(load(response_format={
                "count": {"type": "integer", "items": {"type": "string"}}}))


def test_a_hyphenated_tool_name_is_allowed():
    """Real servers use them — `tavily-search` — and the wire pattern always did.
    Rejecting them meant a tool that exists couldn't be declared."""
    raw = load()
    raw["mcp"][0]["tools"] = [{"tool": "tavily-search"}]
    cfg = AgentConfig.model_validate(raw)
    assert cfg.mcp[0].tools[0].tool == "tavily-search"
    assert "-" in cfg.all_tools[0]


# ── spawning ────────────────────────────────────────────────────────────────


def test_spawns_defaults_to_none_allowed():
    """Silence forbids here, unlike `allowed_capabilities` where it permits. A
    child is a governed unit with a budget of its own, so the risk of a default
    runs the other way."""
    assert AgentConfig.model_validate(load()).spawns == []


def test_an_agent_cannot_spawn_itself():
    """Each child gets a fresh budget, so the parent's ceiling doesn't bound the
    recursion — the config is the only place this can be caught."""
    with pytest.raises(ValidationError, match="itself"):
        AgentConfig.model_validate(load(spawns=["refund-triage"]))


def test_spawns_rejects_duplicates():
    with pytest.raises(ValidationError, match="same agent twice"):
        AgentConfig.model_validate(load(spawns=["outreach", "outreach"]))


def test_starting_a_child_can_be_gated():
    """Spawning commits money nobody has approved, so it belongs in the same
    vocabulary as the other harness tools an operator can put a signature on."""
    cfg = AgentConfig.model_validate(
        load(spawns=["outreach"], gate={"tools": ["start_async_task"]}))
    assert "start_async_task" in cfg.gate.tools
