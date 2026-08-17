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
    assert cfg.gated_tools == ["stripe.create_refund"]
    assert "zendesk.get_ticket" in cfg.inline_tools
    assert "stripe.create_refund" not in cfg.inline_tools
    assert cfg.fail_fast_tools == {"zendesk.get_ticket", "stripe.create_refund"}
    assert len(cfg.all_tools) == 7


def test_invoke_mode_is_derived():
    cfg = AgentConfig.model_validate(load())
    assert cfg.invoke_mode == "queue"  # declares inputs -> discrete tasks

    raw = load()
    raw["inputs"] = {}
    raw["objective"] = "Look for anything that needs attention."
    raw["outcome"]["approval"]["note"] = "no refs"
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

    def test_note_is_checked_too(self):
        raw = load()
        raw["outcome"]["approval"]["note"] = "Ticket {{ inputs.nope }}"
        with pytest.raises(ValidationError, match="not a declared input"):
            AgentConfig.model_validate(raw)


class TestGates:
    def test_gated_tool_without_approval_block_rejected(self):
        """A mutating tool with no gate configured is a silent hole."""
        raw = load()
        raw["outcome"].pop("approval")
        raw["outcome"]["deliverable_approval"] = "never"
        with pytest.raises(ValidationError, match="outcome.approval is required"):
            AgentConfig.model_validate(raw)

    def test_approval_block_with_nothing_gating_rejected(self):
        """Dead config that reads as protection is worse than no config."""
        raw = load()
        raw["outcome"]["deliverable_approval"] = "never"
        for server in raw["mcp"]:
            for tool in server["tools"]:
                tool.pop("approval", None)
        with pytest.raises(ValidationError, match="nothing gates"):
            AgentConfig.model_validate(raw)

    def test_deliverable_approval_alone_requires_block(self):
        raw = load()
        for server in raw["mcp"]:
            for tool in server["tools"]:
                tool.pop("approval", None)
        raw["outcome"]["deliverable_approval"] = "always"
        cfg = AgentConfig.model_validate(raw)
        assert cfg.gated_tools == []


class TestReservedFields:
    @pytest.mark.parametrize("name", ["propose", "ask_human"])
    def test_reserved_deliverable_field_rejected(self, name):
        raw = load()
        raw["outcome"]["deliverable"][name] = {"type": "string"}
        with pytest.raises(ValidationError, match="reserved"):
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
        raw["mcp"][1]["url"] = "http://mcp.stripe.com"
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


class TestAskHuman:
    def test_posture_becomes_prompt_guidance(self):
        """A posture, not a threshold — 'how cautious to be' carries the reason,
        which a float cannot."""
        cfg = AgentConfig.model_validate(load())
        assert cfg.outcome.ask_human.when == "eagerly"
        assert "When in doubt, ask" in cfg.outcome.ask_human.guidance

    def test_how_many_questions_is_a_limit_not_a_behaviour(self):
        """`when` shapes a tendency and lives with the agent; how many questions it
        may ask is a limit and lives in runtime.yaml with the other limits."""
        raw = load()
        raw["outcome"]["ask_human"]["after_iterations"] = 4
        with pytest.raises(ValidationError, match="Extra inputs"):
            AgentConfig.model_validate(raw)

    def test_default_posture_is_balanced(self):
        raw = load()
        del raw["outcome"]["ask_human"]["when"]
        cfg = AgentConfig.model_validate(raw)
        assert cfg.outcome.ask_human.when == "balanced"

    def test_confidence_is_not_a_knob(self):
        """Self-reported confidence is poorly calibrated — a confidently wrong
        answer reports high — so it is wired to nothing."""
        raw = load()
        raw["outcome"]["ask_human"]["below_confidence"] = 0.7
        with pytest.raises(ValidationError, match="Extra inputs"):
            AgentConfig.model_validate(raw)


class TestMemory:
    def test_example_declares_audit_memory(self):
        cfg = AgentConfig.model_validate(load())
        assert cfg.memory.from_audit.rejections == 10

    def test_rejections_without_gates_rejected(self):
        """An agent nothing gates is never rejected, so recalling rejections would
        read as memory it doesn't have."""
        raw = load()
        raw["outcome"]["deliverable_approval"] = "never"
        raw["outcome"].pop("approval")
        for server in raw["mcp"]:
            for tool in server["tools"]:
                tool.pop("approval", None)
        raw["memory"]["from_audit"]["answers"] = 0
        with pytest.raises(ValidationError, match="never rejected"):
            AgentConfig.model_validate(raw)

    def test_answers_without_ask_human_rejected(self):
        raw = load()
        raw["outcome"].pop("ask_human")
        with pytest.raises(ValidationError, match="cannot ask"):
            AgentConfig.model_validate(raw)

    def test_memory_is_optional(self):
        raw = load()
        raw.pop("memory")
        assert AgentConfig.model_validate(raw).memory is None


class TestSchedule:
    def test_periodic_agent(self):
        raw = load()
        raw.pop("inputs")
        raw["objective"] = "Look for anything that needs attention."
        raw["outcome"]["approval"]["note"] = "no refs"
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
        raw["outcome"]["approval"]["note"] = "no refs"
        raw["schedule"] = {"every": spec}
        assert AgentConfig.model_validate(raw).schedule.every_seconds == seconds

    def test_a_bad_duration_says_what_to_write(self):
        raw = load()
        raw.pop("inputs")
        raw["objective"] = "Look."
        raw["outcome"]["approval"]["note"] = "no refs"
        raw["schedule"] = {"every": "every 15 minutes"}
        with pytest.raises(ValidationError, match="use 30s, 15m, 1h, 7d"):
            AgentConfig.model_validate(raw)

    def test_schedule_and_inputs_are_mutually_exclusive(self):
        """A periodic run has nobody to supply a ticket id."""
        raw = load(schedule={"every": "15m"})
        with pytest.raises(ValidationError, match="mutually exclusive"):
            AgentConfig.model_validate(raw)
