import shutil
from pathlib import Path

import pytest
import yaml

from charter.config.loader import ConfigError, load_agent, load_project

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def project(tmp_path) -> Path:
    """A writable copy of examples/, so tests can break it."""
    dst = tmp_path / "project"
    shutil.copytree(EXAMPLES, dst)
    return dst


def edit(path: Path, mutate):
    raw = yaml.safe_load(path.read_text())
    mutate(raw)
    path.write_text(yaml.safe_dump(raw))


def test_examples_load():
    proj = load_project(EXAMPLES / "worker.yaml")
    assert set(proj.agents) == {"refund-triage", "ticket-summarizer"}
    assert sorted(proj.agents["ticket-summarizer"].versions) == [1, 2]
    assert proj.agents["refund-triage"].latest.version == 1


def test_agent_bundle_holds_every_version(project):
    bundle = load_agent(project / "ticket-summarizer")
    assert bundle.versions[1].model == "claude-haiku-4-5"
    assert bundle.versions[2].model == "claude-sonnet-5"
    assert bundle.latest.version == 2


def test_contrasting_agent_is_coalesce():
    """No inputs -> 'something changed, go look' -> coalescing is correct."""
    bundle = load_agent(EXAMPLES / "ticket-summarizer")
    assert bundle.latest.invoke_mode == "coalesce"
    assert bundle.latest.gated_tools == []


class TestAgentCrossFile:
    def test_filename_version_mismatch(self, project):
        edit(project / "refund-triage" / "v1.yaml", lambda r: r.update(version=2))
        with pytest.raises(ConfigError, match="filename says 1"):
            load_agent(project / "refund-triage")

    def test_name_directory_mismatch(self, project):
        edit(project / "refund-triage" / "v1.yaml", lambda r: r.update(name="refunds"))
        with pytest.raises(ConfigError, match="does not match directory"):
            load_agent(project / "refund-triage")

    def test_runtime_agent_mismatch(self, project):
        edit(project / "refund-triage" / "runtime.yaml", lambda r: r.update(agent="other-agent"))
        with pytest.raises(ConfigError, match="runtime.yaml: agent"):
            load_agent(project / "refund-triage")

    def test_tool_call_limit_for_undeclared_tool(self, project):
        def mutate(raw):
            raw["per_run"]["tool_call_limits"][0]["tool"] = "stripe.nonexistent"
        edit(project / "refund-triage" / "runtime.yaml", mutate)
        with pytest.raises(ConfigError, match="no version of this agent declares"):
            load_agent(project / "refund-triage")

    def test_lifecycle_rule_for_undeclared_tool(self, project):
        def mutate(raw):
            raw["rules"][-1]["when"]["tool"] = "stripe.nonexistent"
        edit(project / "refund-triage" / "lifecycle.yaml", mutate)
        with pytest.raises(ConfigError, match="no version of this agent declares"):
            load_agent(project / "refund-triage")

    def test_set_version_target_missing_on_disk(self, project):
        def mutate(raw):
            raw["rules"][2]["then"]["set_version"]["target"] = 7
        edit(project / "refund-triage" / "lifecycle.yaml", mutate)
        with pytest.raises(ConfigError, match="no v7.yaml"):
            load_agent(project / "refund-triage")

    def test_runtime_file_is_optional(self, project):
        """The smallest useful agent is one v1.yaml. An agent always has a ceiling;
        you just shouldn't need a second file to get one."""
        (project / "refund-triage" / "runtime.yaml").unlink()
        bundle = load_agent(project / "refund-triage")
        assert bundle.runtime_defaulted
        assert bundle.runtime.per_run.max_cost_usd == 1.00

    def test_lifecycle_file_is_optional(self, project):
        (project / "refund-triage" / "lifecycle.yaml").unlink()
        assert load_agent(project / "refund-triage").lifecycle is None

    def test_a_single_version_file_is_a_valid_agent(self, tmp_path):
        d = tmp_path / "minimal"
        d.mkdir()
        (d / "v1.yaml").write_text("""
apiVersion: charter/v1
kind: AgentConfig
name: minimal
version: 1
model: claude-haiku-4-5
objective: Say something useful.
outcome:
  deliverable:
    answer: { type: string }
""")
        bundle = load_agent(d)
        assert bundle.runtime_defaulted and bundle.lifecycle is None
        assert bundle.latest.invoke_mode == "coalesce"

    def test_problems_are_collected_not_raised_one_at_a_time(self, project):
        edit(project / "refund-triage" / "v1.yaml", lambda r: r.update(name="wrong", version=9))
        with pytest.raises(ConfigError) as exc:
            load_agent(project / "refund-triage")
        assert len(exc.value.problems) >= 2


class TestProjectCrossFile:
    def test_served_agent_directory_missing(self, project):
        shutil.rmtree(project / "ticket-summarizer")
        with pytest.raises(ConfigError, match="does not exist"):
            load_project(project / "worker.yaml")

    def test_served_version_missing(self, project):
        def mutate(raw):
            raw["serves"][0]["versions"] = [1, 3]
        edit(project / "worker.yaml", mutate)
        with pytest.raises(ConfigError, match=r"v3.yaml does not exist"):
            load_project(project / "worker.yaml")

    def test_rollback_target_not_served_is_the_outage_check(self, project):
        """ticket-summarizer can set_version to 1; a worker serving only v2 would
        leave the control plane dispatching operations nobody can handle."""
        def mutate(raw):
            for s in raw["serves"]:
                if s["agent"] == "ticket-summarizer":
                    s["versions"] = [2]
        edit(project / "worker.yaml", mutate)
        with pytest.raises(ConfigError, match="would strand the agent"):
            load_project(project / "worker.yaml")

    def test_route_to_unknown_channel(self, project):
        def mutate(raw):
            raw["notifications"]["routes"][0]["channel"] = "nowhere"
        edit(project / "worker.yaml", mutate)
        with pytest.raises(ConfigError, match="not defined"):
            load_project(project / "worker.yaml")


class TestRouting:
    def test_agent_route_wins_over_default(self):
        proj = load_project(EXAMPLES / "worker.yaml")
        notif = proj.manifest.notifications
        assert notif.resolve("refund-triage", "approval_requested").name == "finance"
        assert notif.resolve("ticket-summarizer", "approval_requested").name == "oncall"
