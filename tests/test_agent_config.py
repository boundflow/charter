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


def test_read_loop_recovery_is_opt_in_for_declared_nonwrite_tools():
    assert AgentConfig.model_validate(load()).repeated_read_tools == []
    read = 'support__get_ticket'
    assert AgentConfig.model_validate(load(repeated_read_tools=[read])).repeated_read_tools == [read]
    with pytest.raises(ValidationError, match='declared inline tools'):
        AgentConfig.model_validate(load(repeated_read_tools=['unknown']))
    child = {'name': 'writer', 'description': 'Repair', 'prompt': 'Repair'}
    with pytest.raises(ValidationError, match='must not include write tools'):
        AgentConfig.model_validate(load(repeated_read_tools=[read], subagents=[child],
            subagent_file_ownership={'write_tools': {read: ['path']}}))


def test_delegation_prerequisites_are_opt_in_and_require_visible_declared_tools():
    assert AgentConfig.model_validate(load()).subagent_required_tools == []
    cfg = AgentConfig.model_validate(load(subagent_required_tools=['support__get_ticket']))
    assert cfg.subagent_required_tools == ['support__get_ticket']
    with pytest.raises(ValidationError, match='declared tools'):
        AgentConfig.model_validate(load(subagent_required_tools=['unknown']))
    with pytest.raises(ValidationError, match='available to the coordinator'):
        AgentConfig.model_validate(load(subagent_required_tools=['support__get_ticket'], coordinator_tools=[]))
    assert AgentConfig.model_validate(load()).subagent_review_tools == []
    assert AgentConfig.model_validate(load(subagent_review_tools=['support__get_ticket'])).subagent_review_tools
    with pytest.raises(ValidationError, match='declared tools'):
        AgentConfig.model_validate(load(subagent_review_tools=['unknown']))
    with pytest.raises(ValidationError, match='available to the coordinator'):
        AgentConfig.model_validate(load(subagent_review_tools=['support__get_ticket'], coordinator_tools=[]))


def test_file_ownership_requires_declared_write_tools_and_specialists():
    assert AgentConfig.model_validate(load()).subagent_file_ownership is None
    options = {'write_tools': {'support__create_refund': ['path', 'file_path']}, 'max_files': 4}
    child = {'name': 'writer', 'description': 'Assigned writes', 'prompt': 'Write assigned files'}
    cfg = AgentConfig.model_validate(load(subagents=[child], subagent_file_ownership=options))
    assert cfg.subagent_file_ownership.max_files == 4
    with pytest.raises(ValidationError, match='configured specialists'):
        AgentConfig.model_validate(load(subagent_file_ownership=options))
    with pytest.raises(ValidationError, match='declared tools'):
        AgentConfig.model_validate(load(subagents=[child], subagent_file_ownership={'write_tools': {'unknown': ['path']}}))
    for invalid in [True, '4', 0, 101]:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(load(subagents=[child], subagent_file_ownership={**options, 'max_files': invalid}))
    for fields in [[], ['path', 'path'], ['not.a.field']]:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(load(subagents=[child], subagent_file_ownership={'write_tools': {'support__create_refund': fields}}))


def test_specialist_read_prerequisites_require_usable_nonwrite_tools():
    assert AgentConfig.model_validate(load()).subagent_required_read_tools == []
    read = 'support__get_ticket'
    write = 'support__create_refund'
    child = {'name': 'writer', 'description': 'Repair', 'prompt': 'Repair', 'tools': [read]}
    options = dict(subagents=[child], subagent_required_read_tools=[read],
                   subagent_file_ownership={'write_tools': {write: ['path']}})
    assert AgentConfig.model_validate(load(**options)).subagent_required_read_tools == [read]
    with pytest.raises(ValidationError, match='requires subagent_file_ownership'):
        AgentConfig.model_validate(load(**{**options, 'subagent_file_ownership': None}))
    with pytest.raises(ValidationError, match='declared inline tools'):
        AgentConfig.model_validate(load(**{**options, 'subagent_required_read_tools': ['unknown']}))
    with pytest.raises(ValidationError, match='declared inline tools'):
        AgentConfig.model_validate(load(**{**options, 'subagent_required_read_tools': [write]}))
    with pytest.raises(ValidationError, match='every specialist'):
        AgentConfig.model_validate(load(**{**options, 'subagents': [{**child, 'tools': []}], 'coordinator_tools': []}))
    assert AgentConfig.model_validate(load(**{**options, 'subagents': [{**child, 'tools': []}]}))
    with pytest.raises(ValidationError, match='must not include write tools'):
        AgentConfig.model_validate(load(**{**options, 'subagent_file_ownership': {'write_tools': {read: ['path']}}}))


def test_tool_format_recovery_is_opt_in_and_bounded():
    assert AgentConfig.model_validate(load()).normalize_qwen_tool_fragments is False
    assert AgentConfig.model_validate(load(normalize_qwen_tool_fragments=True)).normalize_qwen_tool_fragments is True
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(load(normalize_qwen_tool_fragments='true'))
    assert AgentConfig.model_validate(load()).tool_format_retries == 0
    assert AgentConfig.model_validate(load(tool_format_retries=2)).tool_format_retries == 2
    for invalid in [-1, 4, True, '2']:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(load(tool_format_retries=invalid))


def test_derived_views():
    cfg = AgentConfig.model_validate(load())
    assert cfg.gated_tools == ["support__create_refund"]
    assert "support__get_ticket" in cfg.inline_tools
    assert "support__create_refund" not in cfg.inline_tools
    assert cfg.fail_fast_tools == {"support__get_ticket", "support__create_refund"}
    assert len(cfg.all_tools) == 3


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


def test_tool_bindings_require_declared_tools_and_inputs():
    valid = {'support__get_ticket': {'ticket_id': 'ticket_id'}}
    assert AgentConfig.model_validate(load(tool_input_bindings=valid)).tool_input_bindings == valid
    with pytest.raises(ValidationError, match='unknown declared tool'):
        AgentConfig.model_validate(load(tool_input_bindings={'invented': {'id': 'ticket_id'}}))
    with pytest.raises(ValidationError, match='needs a declared input'):
        AgentConfig.model_validate(load(tool_input_bindings={'support__get_ticket': {'id': 'missing'}}))


def test_coordinator_tool_subset_rejects_unknown_tools():
    assert AgentConfig.model_validate(load(coordinator_tools=[])).coordinator_tools == []
    with pytest.raises(ValidationError, match='coordinator_tools'):
        AgentConfig.model_validate(load(coordinator_tools=['invented']))


def test_subagent_context_is_explicit_and_uses_declared_inputs():
    assert AgentConfig.model_validate(load()).subagent_context_inputs == []
    assert AgentConfig.model_validate(load(subagent_context_inputs=['ticket_id'])).subagent_context_inputs == ['ticket_id']
    with pytest.raises(ValidationError, match='subagent_context_inputs'):
        AgentConfig.model_validate(load(subagent_context_inputs=['undeclared_secret']))


def test_coordinator_context_is_explicit_and_uses_declared_inputs():
    assert AgentConfig.model_validate(load()).coordinator_context_inputs == []
    assert AgentConfig.model_validate(load(coordinator_context_inputs=['ticket_id'])).coordinator_context_inputs == ['ticket_id']
    with pytest.raises(ValidationError, match='coordinator_context_inputs'):
        AgentConfig.model_validate(load(coordinator_context_inputs=['undeclared_secret']))


def test_builtin_exclusions_are_opt_in_and_do_not_accept_mcp_names():
    assert AgentConfig.model_validate(load()).builtin_tools_exclude == []
    assert AgentConfig.model_validate(load(builtin_tools_exclude=['write_file'])).builtin_tools_exclude == ['write_file']
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(load(builtin_tools_exclude=['support__get_ticket']))


class TestTemplating:
    def test_initial_message_validates_declared_inputs(self):
        with pytest.raises(ValidationError, match='initial_message.*not a declared input'):
            AgentConfig.model_validate(load(initial_message='{{ inputs.missing }}'))
        cfg = AgentConfig.model_validate(load(initial_message='Ticket {{ inputs.ticket_id }}'))
        assert cfg.initial_message == 'Ticket {{ inputs.ticket_id }}'

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
        raw["mcp"][0]["command"] = "python"
        with pytest.raises(ValidationError, match="exactly one of"):
            AgentConfig.model_validate(raw)

    def test_neither_command_nor_url_rejected(self):
        raw = load()
        raw["mcp"][0].pop("url")
        with pytest.raises(ValidationError, match="exactly one of"):
            AgentConfig.model_validate(raw)

    def test_http_url_rejected(self):
        """Loopback is the exception; anything else over http carries a token in
        cleartext."""
        raw = load()
        raw["mcp"][0]["url"] = "http://mcp.example.com"
        with pytest.raises(ValidationError, match="https"):
            AgentConfig.model_validate(raw)

    def test_env_value_rejected(self):
        """`env` takes variable NAMES — this file is committed and immutable, so a
        literal secret here would live forever."""
        raw = load()
        raw["mcp"][0]["env"] = ["sk_live_abc123"]
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


class TestAuthority:
    """What the agent may reach, declared in runtime.yaml rather than the version.

    It used to live on AgentConfig, which was wrong twice over: tightening what an
    agent can touch had to wait for a release, and once versions became artifacts
    the value was sealed into one *and* compiled into the policy — where the policy
    copy is the one that binds and the sealed one does nothing.
    """

    def policy(self, **authority):
        from charter.config.runtime import RuntimePolicyFile
        return RuntimePolicyFile.model_validate({
            "apiVersion": "charter/v1", "kind": "RuntimePolicy",
            "agent": "refund-triage", "per_run": {"max_llm_calls": 5},
            "authority": authority})

    def test_paths_must_be_absolute(self):
        with pytest.raises(ValidationError, match="must start with"):
            self.policy(file_rules=[
                {"operations": ["write"], "paths": ["secrets/**"], "mode": "deny"}])

    def test_traversal_rejected(self):
        with pytest.raises(ValidationError, match=r"must not contain"):
            self.policy(file_rules=[
                {"operations": ["write"], "paths": ["/data/../etc"], "mode": "deny"}])

    def test_allow_is_the_default(self):
        rules = self.policy(file_rules=[
            {"operations": ["read"], "paths": ["/data/**"]}]).authority.file_rules
        assert rules[0].mode == "allow"

    def test_an_interrupt_rule_is_visible_to_the_approval_warning(self):
        """Same question `gated_tools` asks, so it wants the same warning: a stop
        for a human that nobody is told about is just a slow timeout."""
        rules = self.policy(file_rules=[
            {"operations": ["write"], "paths": ["/prod/**"], "mode": "interrupt"}
        ]).authority.file_rules
        assert any(r.mode == "interrupt" for r in rules)

    def test_an_empty_allowlist_means_unrestricted(self):
        """Not "nothing allowed". A field that silently forbade everything the
        moment someone added it would be a bad default."""
        assert self.policy().authority.allowed_capabilities == []

    def test_it_matches_the_harness_vocabulary(self):
        """deepagents' own words, so a cap and a permission cannot disagree about
        what `write` means."""
        assert self.policy(allowed_capabilities=["read", "write", "execute", "spawn"])
        with pytest.raises(ValidationError):
            self.policy(allowed_capabilities=["delete"])


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


def policy_with(**authority):
    from charter.config.runtime import RuntimePolicyFile
    return RuntimePolicyFile.model_validate({
        "apiVersion": "charter/v1", "kind": "RuntimePolicy",
        "agent": "leads-finder", "per_run": {"max_llm_calls": 5},
        "authority": authority})


def test_coordinator_inspection_can_be_deferred_without_deferring_prerequisites():
    options = {'coordinator_after_handoff_tools': ['support__get_ticket'],
               'subagents': [{'name': 'worker', 'description': 'Implement a slice'}]}
    assert AgentConfig.model_validate(load(**options)).coordinator_after_handoff_tools == ['support__get_ticket']
    assert AgentConfig.model_validate(load()).coordinator_after_handoff_tools == []
    for change, message in [({'coordinator_after_handoff_tools': ['unknown']}, 'declared tools'),
                            ({'subagents': []}, 'configured specialists'),
                            ({'subagent_required_tools': ['support__get_ticket']}, 'prerequisites'),
                            ({'subagent_review_tools': ['support__get_ticket']}, 'prerequisites'),
                            ({'coordinator_tools': []}, 'available to the coordinator')]:
        with pytest.raises(ValidationError, match=message):
            AgentConfig.model_validate(load(**{**options, **change}))


def test_spawning_defaults_to_none_allowed():
    """Silence forbids here, unlike `allowed_capabilities` where it permits. A
    child is a governed unit with a budget of its own, so the risk of a default
    runs the other way."""
    assert policy_with().authority.allowed_spawns == []


def test_an_agent_cannot_spawn_itself():
    """Each child gets a fresh budget, so the parent's ceiling doesn't bound the
    recursion — policy is the only place it can be caught."""
    with pytest.raises(ValidationError, match="itself"):
        policy_with(allowed_spawns=["leads-finder"])


def test_spawns_rejects_duplicates():
    with pytest.raises(ValidationError, match="same agent twice"):
        policy_with(allowed_spawns=["outreach", "outreach"])


def test_starting_a_child_can_be_gated():
    """Spawning commits money nobody has approved, so it stays in the same
    vocabulary as the other harness tools an operator can put a signature on —
    and that one is versioned, because it changes what stops for a human."""
    cfg = AgentConfig.model_validate(load(gate={"tools": ["start_async_task"]}))
    assert "start_async_task" in cfg.gate.tools


class TestGatedToolsExplainThemselves:
    """A gated call reaches a person with the agent's case for it.

    The harness hands Charter a tool call and nothing else, so a gate could only
    ever show the arguments. `why` is Charter's, added to the schema the model
    sees and removed before the server is called, which never declared it.
    """

    def _tool(self, coroutine=None, func=None, schema=None):
        from types import SimpleNamespace
        return SimpleNamespace(
            name="support__create_refund", coroutine=coroutine, func=func,
            args_schema=schema if schema is not None else {
                "type": "object",
                "properties": {"charge_id": {"type": "string"}},
                "required": ["charge_id"]})

    def test_the_model_is_asked_for_it(self):
        from charter.mcp.client import _explained
        tool = _explained(self._tool(func=lambda **kw: kw))
        assert "justification" in tool.args_schema["properties"]
        assert "justification" in tool.args_schema["required"]

    def test_the_server_never_sees_it(self):
        """It is not a parameter the tool declared, so passing it through would
        fail the call."""
        from charter.mcp.client import _explained
        seen = {}
        tool = _explained(self._tool(func=lambda **kw: seen.update(kw)))
        tool.func(charge_id="ch_1", justification="charged twice")
        assert seen == {"charge_id": "ch_1"}

    def test_it_is_added_once(self):
        from charter.mcp.client import _explained
        tool = _explained(_explained(self._tool(func=lambda **kw: kw)))
        assert tool.args_schema["required"].count("justification") == 1

    def test_a_tool_can_decline_to_be_asked(self):
        """`justify: false` for a call whose arguments already say everything."""
        from charter.config.agent import AgentConfig
        raw = load()
        raw["mcp"][0]["tools"] = [{"tool": "create_refund", "approval": "always",
                                   "justify": False}]
        cfg = AgentConfig.model_validate(raw)
        spec = next(t for t in cfg.mcp[0].tools if t.tool == "create_refund")
        assert spec.justify is False

    def test_it_is_asked_for_by_default(self):
        from charter.config.agent import AgentConfig
        raw = load()
        raw["mcp"][0]["tools"] = [{"tool": "create_refund", "approval": "always"}]
        cfg = AgentConfig.model_validate(raw)
        assert next(t for t in cfg.mcp[0].tools if t.tool == "create_refund").justify
