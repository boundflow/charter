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
            raw["per_run"]["tool_call_limits"][0]["tool"] = "stripe__nonexistent"
        edit(project / "refund-triage" / "runtime.yaml", mutate)
        with pytest.raises(ConfigError, match="no version of this agent declares"):
            load_agent(project / "refund-triage")

    def test_lifecycle_rule_for_undeclared_tool(self, project):
        def mutate(raw):
            raw["rules"][-1]["when"]["tool"] = "stripe__nonexistent"
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


class TestBadPaths:
    """Pointing charter at the wrong directory should say so, not surface a pydantic
    error about a field the user never wrote."""

    def test_a_directory_that_is_not_an_agent(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        with pytest.raises(ConfigError, match="not an agent directory"):
            load_agent(tmp_path)

    def test_dot_resolves_to_a_real_name(self, tmp_path, monkeypatch):
        """Path('.').name is '', which reached pydantic as an agent named ''."""
        d = tmp_path / "my-agent"
        d.mkdir()
        (d / "v1.yaml").write_text("""
apiVersion: charter/v1
kind: AgentConfig
name: my-agent
version: 1
model: claude-haiku-4-5
objective: Do the thing.
""")
        monkeypatch.chdir(d)
        bundle = load_agent(Path("."))
        assert bundle.name == "my-agent"
        assert bundle.runtime.agent == "my-agent"


class TestRunFilters:
    """Narrowing a run history. The API returns everything with no server-side
    filter, so this happens locally — correct, but see the pagination note in
    DESIGN.md before anyone has a month of runs."""

    class R:
        def __init__(self, outcome=None, status="completed", ago_hours=0):
            import datetime as dt
            from types import SimpleNamespace
            self.run_outcome = SimpleNamespace(value=outcome) if outcome else None
            self.status = SimpleNamespace(value=status)
            self.created_at = (dt.datetime.now(dt.timezone.utc)
                               - dt.timedelta(hours=ago_hours))

    def runs(self):
        return [self.R("successful"), self.R("customer_marked_failure"),
                self.R("operation_timeout", ago_hours=48), self.R("successful", ago_hours=48)]

    def test_failed_catches_every_unsuccessful_outcome(self):
        from charter.cli import _filter_runs
        out = _filter_runs(self.runs(), failed=True, status=None, since=None)
        assert {r.run_outcome.value for r in out} == {
            "customer_marked_failure", "operation_timeout"}

    def test_status_is_exact(self):
        from charter.cli import _filter_runs
        out = _filter_runs(self.runs(), failed=False, status="operation_timeout", since=None)
        assert len(out) == 1

    def test_since_accepts_durations_and_dates(self):
        from charter.cli import _filter_runs
        assert len(_filter_runs(self.runs(), failed=False, status=None, since="24h")) == 2
        assert len(_filter_runs(self.runs(), failed=False, status=None, since="7d")) == 4

    def test_filters_compose(self):
        from charter.cli import _filter_runs
        out = _filter_runs(self.runs(), failed=True, status=None, since="24h")
        assert [r.run_outcome.value for r in out] == ["customer_marked_failure"]


class TestInstructions:
    """Instruction documents live in v<N>/ beside v<N>.yaml, so they're versioned by
    the same rule the config is — `set_version: 1` restores the exact prompt v1 ran
    with, and editing one in place is the same mistake as editing its yaml."""

    def _agent(self, root, *, version=1, skills=None):
        d = root / "policy-agent"
        d.mkdir(exist_ok=True)
        (d / f"v{version}.yaml").write_text(f"""
apiVersion: charter/v1
kind: AgentConfig
name: policy-agent
version: {version}
model: claude-haiku-4-5
objective: Decide something.
""")
        for name, text in (skills or {}).items():
            skill = d / f"v{version}" / "skills" / name
            skill.mkdir(parents=True, exist_ok=True)
            (skill / "SKILL.md").write_text(text)
        return d

    def test_a_skills_directory_is_found_without_being_declared(self, tmp_path):
        """No manifest: what is on disk is what ships, so nothing can drift."""
        d = self._agent(tmp_path, skills={"refund-policy": "Refund duplicates only."})
        bundle = load_agent(d)
        assert bundle.skills[1] == d / "v1" / "skills"

    def test_each_version_keeps_its_own_skills(self, tmp_path):
        """The point of the layout: v1 keeps what it shipped with, so a rollback
        restores an agent that still exists."""
        d = self._agent(tmp_path, version=1, skills={"policy": "the old rule"})
        self._agent(tmp_path, version=2, skills={"policy": "the new rule"})
        bundle = load_agent(d)
        assert (bundle.skills[1] / "policy" / "SKILL.md").read_text() == "the old rule"
        assert (bundle.skills[2] / "policy" / "SKILL.md").read_text() == "the new rule"

    def test_an_agent_without_skills_is_fine(self, tmp_path):
        assert load_agent(self._agent(tmp_path)).skills == {}

    def test_a_skill_without_skill_md_is_caught_at_validate(self, tmp_path):
        """SKILL.md is what identifies a skill to the harness — a directory without
        one is silently ignored at runtime, which is worse than failing here."""
        d = self._agent(tmp_path, skills={"policy": "fine"})
        (d / "v1" / "skills" / "empty").mkdir()
        with pytest.raises(ConfigError, match="no SKILL.md"):
            load_agent(d)

    def test_an_empty_skills_directory_is_a_mistake(self, tmp_path):
        d = self._agent(tmp_path)
        (d / "v1" / "skills").mkdir(parents=True)
        with pytest.raises(ConfigError, match="holds no skill directories"):
            load_agent(d)

class TestChannelKinds:
    """Three body shapes, one signed POST. Chat products reject arbitrary JSON, so
    the envelope has to change even though nothing else does."""

    def _channel(self, **kw):
        from charter.config.worker import Channel
        return Channel(**{"name": "c", "url": "https://x", **kw})

    def test_telegram_needs_a_chat_id(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="requires `chat_id`"):
            self._channel(kind="telegram")

    def test_each_kind_shapes_the_body_it_needs(self):
        from charter.notify import _shape
        payload = {"event": "approval_requested", "agent": "a", "approval_id": "apr_1",
                   "justification": "do the thing"}

        assert _shape(self._channel(kind="webhook"), payload) == payload
        assert set(_shape(self._channel(kind="slack"), payload)) == {"text"}

        tg = _shape(self._channel(kind="telegram", chat_id="42"), payload)
        assert tg["chat_id"] == "42"
        assert "apr_1" in tg["text"]

    def test_every_kind_carries_the_command_that_unblocks_it(self):
        """Whoever reads this on a phone shouldn't have to go find an id."""
        from charter.notify import _shape
        for kind, extra in (("slack", {}), ("telegram", {"chat_id": "42"})):
            body = _shape(self._channel(kind=kind, **extra),
                          {"event": "approval_requested", "agent": "leads-finder",
                           "approval_id": "apr_1", "justification": "x"})
            assert "charter approve apr_1 --agent leads-finder" in body["text"]
