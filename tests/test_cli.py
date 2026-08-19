"""The CLI, against a fake control plane that returns *real* SDK types.

This layer had no tests and produced three bugs: `status` read `info.state` (the
field is `status`), `describe` read a config field that didn't exist yet, and
`agents` read `w.config` from `list_workflows`, which returns the lighter view
with it unset.

All three are the same mistake — guessing at a shape — so the fake here builds
genuine WorkflowInfo / RequestInfo / Run / WorkflowMetrics objects. A wrong field
name is then an AttributeError in the test rather than a traceback in someone's
terminal.
"""
from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager

import pytest
from boundflow import (
    InvokeMode,
    LifecycleState,
    RunOutcome,
    RunStatus,
    WorkflowConfig,
    WorkflowState,
)
from boundflow.control_plane import (
    PendingApproval,
    RequestInfo,
    Run,
    Tenant,
    WorkflowInfo,
    WorkflowMetrics,
)
from typer.testing import CliRunner

from charter import cli

runner = CliRunner()
NOW = dt.datetime(2026, 8, 18, 3, 47, tzinfo=dt.timezone.utc)


def workflow(name="refund-demo", *, workflow_state=WorkflowState.ACTIVE,
             lifecycle_state=LifecycleState.ACTIVE, config=None, pending=None):
    return WorkflowInfo(
        id=f"wf_{name}", workflow_type=name, tenant_id="tnt_1",
        lifecycle_state=lifecycle_state, workflow_state=workflow_state,
        version=1, last_interrupted_request_id="", last_policy_decision_request_id="",
        config=config, pending_approval=pending, pending_input=None)


class FakeCp:
    """Only what the CLI calls. Returns real SDK dataclasses."""

    def __init__(self, workflows=None, request=None, runs=None, metrics=None):
        self.workflows = workflows if workflows is not None else [workflow()]
        self.request = request
        self.runs = runs or []
        self.metrics = metrics or WorkflowMetrics(
            version=1, total_cost_usd=0.0041, run_count=3, total_failures=1,
            total_llm_calls=7, total_latency_seconds=12.0,
            total_approval_rejections=0, tool_failure_counts={})
        self.approved = []

    async def list_tenants(self):
        return [Tenant(id="tnt_1", name="default", tenant_group_id="tg_1",
                       deleted_at=None)]

    async def list_workflows(self):
        # The lighter view: config and pending gates are unset here.
        return [workflow(w.workflow_type, workflow_state=w.workflow_state,
                         lifecycle_state=w.lifecycle_state) for w in self.workflows]

    async def get_workflow(self, workflow_id):
        return next(w for w in self.workflows if w.id == workflow_id)

    async def get_workflow_metrics(self, workflow_id):
        return self.metrics

    async def list_workflow_runs(self, workflow_id):
        return self.runs

    async def get_request_info(self, request_id):
        return self.request

    async def get_agent_runtime_policy(self, workflow_id, agent):
        return {"maxCostUsd": 0.25, "maxLlmCalls": 20}

    async def get_agent_lifecycle_policy(self, workflow_id, agent):
        return {}

    async def get_workflow_lifecycle_policy(self, workflow_id):
        return []

    async def approve_workflow(self, workflow_id, approval_id, actor="", reason=""):
        self.approved.append((approval_id, actor, reason))


@pytest.fixture
def cp(monkeypatch):
    fake = FakeCp()

    @asynccontextmanager
    async def _cp():
        yield fake

    monkeypatch.setattr(cli, "_cp", lambda: _cp())
    monkeypatch.setenv("CHARTER_TENANT", "default")
    return fake


def invoke(*args):
    return runner.invoke(cli.app, list(args))


class TestAgents:
    def test_lists_the_tenants_agents(self, cp):
        cp.workflows = [workflow("refund-demo"), workflow("ticket-sweeper")]
        result = invoke("agents")
        assert result.exit_code == 0, result.output
        assert "refund-demo" in result.output and "ticket-sweeper" in result.output

    def test_survives_the_lighter_view_having_no_config(self, cp):
        """list_workflows returns config=None. Reading it was an AttributeError
        that only appeared against a real server."""
        cp.workflows = [workflow("refund-demo")]
        assert invoke("agents").exit_code == 0

    def test_points_a_parked_agent_at_pending(self, cp):
        cp.workflows = [workflow("refund-demo",
                                 lifecycle_state=LifecycleState.AWAITING_APPROVAL)]
        out = invoke("agents").output
        assert "waiting on a human" in out
        assert "charter pending refund-demo" in out

    def test_points_a_paused_agent_at_resume(self, cp):
        cp.workflows = [workflow("refund-demo", workflow_state=WorkflowState.PAUSED)]
        out = invoke("agents").output
        assert "stopped" in out
        assert "charter resume refund-demo" in out

    def test_says_so_when_there_are_none(self, cp):
        cp.workflows = []
        assert "no agents" in invoke("agents").output


class TestStatus:
    def _request(self, **kw):
        base = dict(request_id="req_1", workflow_id="wf_refund-demo",
                    request_type="invoke", status=RunStatus.COMPLETED,
                    run_outcome=RunOutcome.SUCCESSFUL, failure_reason="",
                    sequence_number=1, created_at=NOW,
                    completed_at=NOW + dt.timedelta(seconds=8),
                    result=None, invoke_context=None, timeout_seconds=60,
                    agent_runtime_policies=None)
        return RequestInfo(**{**base, **kw})

    def test_shows_the_deliverable(self, cp):
        cp.request = self._request(result={"summary": "two need a look",
                                           "needs_attention": 2, "cost_usd": 0.002,
                                           "rounds": 2})
        out = invoke("status", "req_1").output
        assert "two need a look" in out
        assert "successful" in out
        assert "8s" in out          # took, from completed_at - created_at

    def test_shows_a_platform_failure_in_full(self, cp):
        """An uncaught exception never publishes a result, so failure_reason is the
        only record — and it was never printed at all."""
        cp.request = self._request(
            status=RunStatus.FAILED, run_outcome=RunOutcome.UNCAUGHT_OPERATION_EXCEPTION,
            failure_reason="BadRequestError: 400 tools.0.custom.name: bad pattern")
        out = invoke("status", "req_1").output
        assert "tools.0.custom.name" in out

    def test_shows_actions_already_taken_on_a_failure(self, cp):
        """A failed task is not an untouched one."""
        cp.request = self._request(result={
            "failed": True, "reason": "ran out of budget",
            "acts_performed": [{"tool": "desk__create_refund", "args": {"amount": 240}}]})
        out = invoke("status", "req_1").output
        assert "desk__create_refund" in out
        assert "ran out of budget" in out


class TestPendingAndApprove:
    def test_pending_renders_the_gate_and_the_command(self, cp):
        cp.workflows = [workflow(
            "refund-demo", lifecycle_state=LifecycleState.AWAITING_APPROVAL,
            pending=PendingApproval(approval_id="apr_1",
                                    justification="run desk__create_refund\n  amount: 240",
                                    metadata={}, opened_at=NOW, timeout_at=None))]
        out = invoke("pending", "refund-demo").output
        assert "needs approval" in out
        assert "amount: 240" in out
        assert "charter approve apr_1" in out

    def test_pending_says_nothing_is_waiting(self, cp):
        assert "nothing waiting" in invoke("pending", "refund-demo").output

    def test_approve_passes_the_reason(self, cp):
        invoke("approve", "apr_1", "--agent", "refund-demo", "--reason", "confirmed")
        assert cp.approved == [("apr_1", "", "confirmed")]


class TestTasks:
    def test_puts_the_failure_reason_under_its_row(self, cp):
        cp.runs = [Run(request_id="req_1", request_type="invoke",
                       status=RunStatus.FAILED,
                       run_outcome=RunOutcome.UNCAUGHT_OPERATION_EXCEPTION,
                       failure_reason="BadRequestError: 400 - " + "x" * 120,
                       created_at=NOW, completed_at=NOW + dt.timedelta(seconds=2))]
        out = invoke("tasks", "refund-demo").output
        # Whole message, not a column truncated at some number I picked.
        assert "x" * 120 in out


class TestUnappliedAgent:
    def test_every_command_says_apply_first(self, cp):
        cp.workflows = []
        for args in (("describe", "nope"), ("tasks", "nope"), ("pending", "nope")):
            result = invoke(*args)
            assert result.exit_code == 1, args


class TestImport:
    """charter import against the real fixture server — it's the one command whose
    whole job is talking to a server, so a fake would test nothing."""

    def test_drafts_a_block_with_dangerous_tools_gated(self):
        # Run as a real process: CliRunner swaps stdout for a StringIO, and spawning
        # an stdio MCP server needs a real file descriptor.
        import subprocess
        import sys
        from pathlib import Path

        server = Path(__file__).parent / "mcp_fixture_server.py"
        charter = Path(sys.executable).parent / "charter"
        if not charter.exists():
            pytest.skip("charter console script not installed")
        proc = subprocess.run(
            [str(charter), "import", "desk", "--command", sys.executable, "--arg", str(server)],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout + proc.stderr

        # get_ticket is annotated read-only in the fixture; the rest are not.
        assert "- tool: get_ticket   # read_only_hint" in out
        assert "close_ticket" in out and "approval: always" in out
        # Undeclared by default, so "5 available, 2 declared" is visible while
        # authoring rather than in a log line at boot.
        assert "# - tool:" in out
        assert "read-only" in out and "gated" in out

    def test_needs_exactly_one_transport(self):
        assert runner.invoke(cli.app, ["import", "desk"]).exit_code == 1
        assert runner.invoke(cli.app, ["import", "desk", "--url", "https://x",
                                       "--command", "y"]).exit_code == 1


class TestSchema:
    """JSON Schema generation.

    The value here is that the schema tracks the models automatically, so these
    tests assert the properties an editor actually depends on — enums for the
    closed vocabularies, `additionalProperties: false` so a typo is an error
    rather than a silently-ignored key, and descriptions for hover text.
    """

    def test_emits_all_four(self, tmp_path):
        import json

        res = runner.invoke(cli.app, ["schema", "-o", str(tmp_path)])
        assert res.exit_code == 0, res.output

        for kind in ("agent", "runtime", "lifecycle", "worker"):
            doc = json.loads((tmp_path / f"{kind}.schema.json").read_text())
            assert doc["$schema"].startswith("https://json-schema.org/")
            assert doc["$id"].endswith(f"/{kind}.json")
            # Unknown keys must be errors — the whole point is catching typos.
            assert doc["additionalProperties"] is False

    def test_closed_vocabularies_are_enums(self, tmp_path):
        """Someone writing `metric:` should get the six valid values offered."""
        import json

        runner.invoke(cli.app, ["schema", "-o", str(tmp_path)])
        life = json.loads((tmp_path / "lifecycle.schema.json").read_text())
        metric = life["$defs"]["When"]["properties"]["metric"]
        assert set(metric["enum"]) == {
            "num_failures", "cost", "num_llm_calls", "latency",
            "approval_rejections", "tool_failures"}
        # And the enum carries prose, so the dropdown explains itself.
        assert "requires `tool`" in metric["description"]

        agent = json.loads((tmp_path / "agent.schema.json").read_text())
        approval = agent["$defs"]["ToolSpec"]["properties"]["approval"]
        assert {"never", "always"} == {
            v for branch in approval["anyOf"] for v in branch.get("enum", [])}

    def test_authored_fields_carry_hover_text(self, tmp_path):
        """Comments in the source are invisible to the schema, so the fields
        people actually type need `description=` on the Field itself."""
        import json

        runner.invoke(cli.app, ["schema", "-o", str(tmp_path)])
        run = json.loads((tmp_path / "runtime.schema.json").read_text())
        props = run["$defs"]["PerRun"]["properties"]
        for field in ("max_cost_usd", "max_llm_calls", "max_tool_failures"):
            assert props[field].get("description"), f"{field} has no hover text"

    def test_single_kind_to_stdout(self):
        import json

        res = runner.invoke(cli.app, ["schema", "--kind", "agent"])
        assert res.exit_code == 0
        assert json.loads(res.output)["$id"].endswith("/agent.json")

    def test_rejects_unknown_kind_and_missing_destination(self):
        assert runner.invoke(cli.app, ["schema", "--kind", "nope"]).exit_code == 1
        # No --out and no --kind is ambiguous rather than a silent no-op.
        assert runner.invoke(cli.app, ["schema"]).exit_code == 1
