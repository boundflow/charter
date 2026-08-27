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
from dataclasses import replace

import pytest

from pathlib import Path as _P
EXAMPLES = _P(__file__).parent.parent / "examples"
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
    Suspension,
    WorkflowInfo,
    WorkflowMetrics,
)
from typer.testing import CliRunner

from charter import cli

runner = CliRunner()
NOW = dt.datetime(2026, 8, 18, 3, 47, tzinfo=dt.timezone.utc)


def workflow(name="refund-demo", *, workflow_state=WorkflowState.ACTIVE,
             lifecycle_state=LifecycleState.ACTIVE, config=None, pending=None,
             suspension=None):
    return WorkflowInfo(
        id=f"wf_{name}", workflow_type=name, tenant_id="tnt_1",
        lifecycle_state=lifecycle_state, workflow_state=workflow_state,
        version=1, last_interrupted_request_id="", last_policy_decision_request_id="",
        config=config, pending_approval=pending, pending_input=None,
        suspension=suspension)


def held(sid="sus_theirs", reason="incident", stop_current=False, finalized=True):
    """A workflow under an operator hold, ready to drain."""
    from datetime import datetime
    return workflow(
        workflow_state=WorkflowState.SUSPENDED,
        suspension=Suspension(
            suspension_id=sid, reason=reason, stop_current=stop_current,
            requested_at=datetime(2026, 1, 1),
            finalized_at=datetime(2026, 1, 1) if finalized else None))


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
        self.resumed: list = []
        self.suspended: list = []

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

    async def resume_workflow(self, workflow_id, suspension_id):
        self.resumed.append((workflow_id, suspension_id))

    async def suspend_workflow(self, workflow_id, reason="", stop_current_run=False,
                               suspension_id=""):
        self.suspended.append((workflow_id, reason, stop_current_run, suspension_id))
        return suspension_id or "sus_new"


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

    def test_a_failure_shows_what_it_spent_getting_there(self, cp):
        """The reason alone doesn't say whether to raise the ceiling or fix the
        agent. The numbers that stopped it are the thing an operator acts on."""
        cp.request = self._request(result={
            "failed": True, "reason": "ran out of budget",
            "cost_usd": 0.42, "llm_calls": 12, "gates": 1, "seconds": 31.5})
        out = invoke("status", "req_1").output
        assert "ran out of budget" in out
        assert "0.42" in out and "12" in out

    def test_a_successful_result_is_only_the_agent_answer(self, cp):
        """Charter injects nothing into a result it didn't fail, so there is no
        wrapper to unpick and nothing of ours to strip out of the display."""
        cp.request = self._request(result={"summary": "two need a look",
                                           "needs_attention": 2})
        out = invoke("status", "req_1").output
        assert "two need a look" in out
        assert "spent" not in out


class TestPendingAndApprove:
    def test_pending_renders_the_gate_and_the_command(self, cp):
        cp.workflows = [workflow(
            "refund-demo", lifecycle_state=LifecycleState.AWAITING_APPROVAL,
            pending=PendingApproval(approval_id="apr_1",
                                    justification="run desk__create_refund\n  amount: 240",
                                    metadata={}, opened_at=NOW, timeout_at=None))]
        out = invoke("pending", "refund-demo", "--instance", "wf_refun").output
        assert "needs approval" in out
        assert "amount: 240" in out
        assert "charter approve apr_1" in out

    def test_pending_says_nothing_is_waiting(self, cp):
        assert "nothing waiting" in invoke("pending", "refund-demo", "--instance", "wf_refun").output

    def test_approve_passes_the_reason(self, cp):
        invoke("approve", "apr_1", "--agent", "refund-demo", "--instance",
               "wf_refun", "--reason", "confirmed")
        assert cp.approved == [("apr_1", "", "confirmed")]


class TestTasks:
    def test_puts_the_failure_reason_under_its_row(self, cp):
        cp.runs = [Run(request_id="req_1", request_type="invoke",
                       status=RunStatus.FAILED,
                       run_outcome=RunOutcome.UNCAUGHT_OPERATION_EXCEPTION,
                       failure_reason="BadRequestError: 400 - " + "x" * 120,
                       created_at=NOW, completed_at=NOW + dt.timedelta(seconds=2))]
        out = invoke("tasks", "refund-demo", "--instance", "wf_refun").output
        # Whole message, not a column truncated at some number I picked.
        assert "x" * 120 in out


class TestUnappliedAgent:
    def test_every_command_says_apply_first(self, cp):
        cp.workflows = []
        for args in (("describe", "nope"), ("tasks", "nope"), ("pending", "nope")):
            result = invoke(*args)
            assert result.exit_code == 1, args


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


class TestInstances:
    """An agent can have several instances — same config, separate entities, each
    with its own store namespace, budget and lifecycle state. Commands that could
    touch the wrong one refuse to guess."""

    def test_short_ids_are_derived_not_positional(self):
        """An ordinal moves when a neighbour is deleted, silently repointing
        anything that referred to it. A short id never moves."""
        from charter.cli import short
        assert short("7f3a91c2-1234-5678-9abc-def012345678") == "7f3a91c2"

    def test_run_refuses_when_several_instances_exist(self, monkeypatch):
        res = runner.invoke(cli.app, ["run", "refund-triage", "--path", str(EXAMPLES)])
        # Either it asks which, or it fails earlier on config — both are refusals,
        # and neither silently picks one.
        assert res.exit_code != 0

    def test_apply_never_creates_an_instance(self):
        """Creating an entity with state of its own shouldn't be a side effect of
        re-running config in CI."""
        from charter.provisioning import apply as mod
        import inspect
        src = inspect.getsource(mod.apply_agent)
        assert "create_workflow" not in src


# ── the bodies actually run ─────────────────────────────────────────────────


def test_no_command_references_an_option_it_never_declared():
    """`charter answer` raised NameError: instance — it used `instance` to pick the
    workflow but never took the flag. `charter delete` had it too, which is worse,
    because that one deletes.

    Neither was caught by anything: the suite exercises argument parsing and the
    control-plane calls, and a name resolved at runtime inside `go()` is invisible
    to both until a person runs the command. This is a cheap static stand-in for
    running all of them.
    """
    import ast
    import inspect
    from pathlib import Path

    import charter.cli as cli

    source = Path(inspect.getfile(cli)).read_text()
    tree = ast.parse(source)
    module_level = {n.id for stmt in tree.body for n in ast.walk(stmt)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    module_level |= {n.name for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    module_level |= {a.asname or a.name.split(".")[0]
                     for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    module_level |= {a.asname or a.name
                     for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}

    # The options a command forgets to declare, which is the whole failure mode.
    flags = {"instance", "tenant", "agent", "all_", "yes", "actor", "dry_run"}
    problems = []
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        bound = set(params)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
                bound |= {a.arg for a in node.args.args}
        for node in ast.walk(fn):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id in flags and node.id not in bound
                    and node.id not in module_level):
                problems.append(f"{fn.name}: uses {node.id!r}, never declared")

    assert not problems, "\n".join(sorted(set(problems)))


# ── the control plane's record, not our summary of it ───────────────────────


class TestTheHoldIsVisibleAndOwned:
    """A suspension_id says whose hold it is. Charter used to discard the one
    `suspend_workflow` returns and read whatever hold the server reported, so
    `charter resume` released someone else's pause without either operator being
    able to tell afterwards."""

    def test_describe_shows_whose_hold_it_is_and_whether_it_drained(self, cp):
        cp.workflows = [held(sid="sus_abc", reason="incident 4821", finalized=False)]
        result = invoke("describe", "refund-demo", "--instance", "wf_refun")
        # Not just the substring: these once passed while describe crashed on the
        # next section, because the hold prints before the config does.
        assert result.exit_code == 0
        out = result.output
        assert "sus_abc" in out
        assert "incident 4821" in out
        # The question you actually have at 3am, and the reason resume refuses.
        assert "cannot be resumed yet" in out

    def test_resume_will_not_release_a_hold_you_did_not_name(self, cp):
        cp.workflows = [held(sid="sus_theirs", reason="incident")]
        result = invoke("resume", "refund-demo", "--instance", "wf_refun")
        assert result.exit_code == 1
        assert cp.resumed == []
        assert "sus_theirs" in result.output

    def test_resume_releases_the_hold_you_named(self, cp):
        cp.workflows = [held(sid="sus_mine")]
        result = invoke("resume", "refund-demo", "--instance", "wf_refun", "--suspension", "sus_mine")
        assert result.exit_code == 0
        assert cp.resumed == [("wf_refund-demo", "sus_mine")]

    def test_resume_refuses_when_the_hold_is_not_the_one_you_placed(self, cp):
        """Yours was released and another placed since. Releasing this one anyway
        would undo a stranger's pause while you believed you were undoing your own."""
        cp.workflows = [held(sid="sus_theirs")]
        result = invoke("resume", "refund-demo", "--instance", "wf_refun", "--suspension", "sus_mine")
        assert result.exit_code == 1
        assert cp.resumed == []

    def test_reading_the_id_is_the_only_way_through(self, cp):
        """There is no --force. BoundFlow makes suspension_id a required positional
        on `workflow resume`, and nothing enforces ownership anywhere — the id is
        explicitness, not permission. A flag meaning "whoever placed it" would be
        the same call with the deliberation removed."""
        cp.workflows = [held(sid="sus_theirs")]
        assert "--force" not in invoke("resume", "--help").output
        assert invoke("resume", "refund-demo", "--instance", "wf_refun", "--suspension",
                      "sus_theirs").exit_code == 0
        assert cp.resumed == [("wf_refund-demo", "sus_theirs")]

    def test_pause_hands_back_the_id_so_it_can_be_named_later(self, cp):
        out = invoke("pause", "refund-demo", "--instance", "wf_refun").output
        assert "sus_new" in out

    def test_now_escalates_an_existing_hold_rather_than_reporting_it(self, cp):
        """`--now` on an already-paused agent is asking for the run to stop, which a
        graceful pause never did. Reporting "already paused" would make the second,
        more urgent command do strictly less than the first."""
        cp.workflows = [held(sid="sus_abc", stop_current=False, finalized=False)]
        invoke("pause", "refund-demo", "--instance", "wf_refun", "--now")
        # Retargeted, not a second hold — resume has one id to release.
        assert cp.suspended == [("wf_refund-demo", "incident", True, "sus_abc")]

    def test_deletion_pending_is_not_hidden(self, cp):
        from datetime import datetime
        wf = workflow()
        cp.workflows = [replace(wf, deletion_requested_at=datetime(2026, 1, 1))]
        assert "deletion requested" in invoke("describe", "refund-demo", "--instance", "wf_refun").output


def test_no_two_top_level_definitions_share_a_name():
    """A second `def _when` silently replaced the first, so the caller expecting
    full timestamps got clock time and the new function was dead. Python does not
    warn, and neither did anything else here — the same gap that let `charter
    answer` ship a NameError."""
    import ast
    import inspect
    from collections import Counter
    from pathlib import Path

    import charter.cli as cli

    tree = ast.parse(Path(inspect.getfile(cli)).read_text())
    names = Counter(n.name for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    dupes = {n: c for n, c in names.items() if c > 1}
    assert not dupes, f"defined more than once: {dupes}"


def test_every_workflow_field_is_shown_or_deliberately_left_out():
    """The audit that found `describe` rendering 6 of WorkflowInfo's 13 fields,
    turned into something that fails a build.

    Curation is the point of `describe` — it is the screen you read at 3am, not a
    field dump. But curation drifts silently: BoundFlow adds a field, nobody here
    notices, and the command that claims to show everything about an agent quietly
    stops doing so. So a new field has to be either rendered or named below, and
    `--json` reaches the whole record either way.
    """
    import ast
    import dataclasses as dc
    import inspect
    from pathlib import Path

    from boundflow.control_plane import WorkflowInfo

    import charter.cli as cli

    # Deliberately absent from the curated view, with the reason.
    OMITTED = {
        # The agent's name is the header; its id is already shown as "workflow".
        "workflow_type", "id",
        # You addressed this agent by tenant to get here.
        "tenant_id",
        # Shown by `charter status` against a task, where it means something.
        "last_policy_decision_request_id",
    }

    tree = ast.parse(Path(inspect.getfile(cli)).read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "describe")
    body = ast.unparse(fn)

    missing = [f.name for f in dc.fields(WorkflowInfo)
               if f.name not in body and f.name not in OMITTED]
    assert not missing, (
        f"`charter describe` never mentions {missing}. Render them, or add them to "
        f"OMITTED here with the reason.")


def test_json_renders_records_not_reprs(cp):
    """`output()` converts the object it is handed, not ones nested inside it, so a
    composite view has to be flattened first or the JSON carries Python reprs."""
    import json

    cp.workflows = [held(sid="sus_abc")]
    doc = json.loads(invoke("--json", "describe", "refund-demo", "--instance", "wf_refun").output)
    assert doc["workflow"]["suspension"]["suspension_id"] == "sus_abc"
    assert doc["workflow"]["workflow_state"] == "suspended"   # enum, not repr


def test_the_cli_reads_the_manifest_for_its_connection(tmp_path, monkeypatch):
    """`charter worker` used the control_plane block in worker.yaml and every other
    command ignored it, so a key written into config connected the worker and
    nothing else. One place to put it."""
    monkeypatch.setenv("CHARTER_PROJECT", "examples/worker.yaml")

    manifest = cli._manifest()
    assert manifest is not None
    assert manifest.control_plane.endpoint
    # The tenant comes from the file when nothing more specific says otherwise.
    monkeypatch.delenv("CHARTER_TENANT", raising=False)
    assert cli._tenant_name(None) == manifest.control_plane.tenant


def test_a_broken_manifest_says_so_rather_than_silently_using_the_environment(
        tmp_path, monkeypatch, capsys):
    """The file names a control plane. If it cannot be read, using whatever the
    environment had — without a word — is how you act on the wrong one."""
    monkeypatch.setenv("CHARTER_PROJECT", str(tmp_path / "worker.yaml"))
    (tmp_path / "worker.yaml").write_text(
        "apiVersion: charter/v1\nkind: Worker\nname: w\nserves: []\n")

    assert cli._manifest() is None
    assert "could not be read" in capsys.readouterr().out


def test_no_manifest_is_not_an_error(tmp_path, monkeypatch):
    """Most commands work without a checkout — on call you have credentials and an
    agent name, not the repo."""
    monkeypatch.setenv("CHARTER_PROJECT", str(tmp_path / "nothing.yaml"))
    assert cli._manifest() is None


def test_every_command_the_readme_names_exists():
    """A README is the one place a command name goes stale without anything
    failing. Renaming a command leaves the docs confidently wrong, and the person
    who finds out is someone following them for the first time."""
    import re
    from pathlib import Path

    import typer

    names = set()
    for group in (cli.app, ):
        for command in group.registered_commands:
            names.add(command.name or command.callback.__name__.replace("_", "-"))
    for sub in cli.app.registered_groups:
        for command in sub.typer_instance.registered_commands:
            names.add(f"{sub.name} {command.name}")

    documented = set()
    for line in Path("README.md").read_text().splitlines():
        m = re.match(r"^charter ([a-z-]+)(?: ([a-z-]+))?", line.strip())
        if not m:
            continue
        verb, second = m.group(1), m.group(2)
        documented.add(f"{verb} {second}" if f"{verb} {second}" in names else verb)

    missing = sorted(d for d in documented if d not in names)
    assert not missing, f"README names commands that do not exist: {missing}"


class TestAnInstanceIsAlwaysNamed:
    """An agent name addresses a *kind* of instance. The instance is what holds the
    state — the conversation, the budget, the lifecycle history — so a command that
    acts has to say which one, even when there is only one to pick."""

    def test_one_instance_is_still_not_a_default(self, cp):
        result = invoke("describe", "refund-demo")
        assert result.exit_code == 1
        assert "say which" in result.output
        # And it hands over the id, so being explicit is a copy-paste.
        assert "wf_refun" in result.output

    def test_naming_it_works(self, cp):
        assert invoke("describe", "refund-demo", "--instance",
                      "wf_refun").exit_code == 0

    def test_a_prefix_is_enough_until_it_is_ambiguous(self, cp):
        cp.workflows = [workflow(), workflow()]
        result = invoke("describe", "refund-demo", "--instance", "wf_")
        assert result.exit_code == 1
        assert "matches 2 instances" in result.output

    def test_an_unknown_id_lists_the_real_ones(self, cp):
        result = invoke("describe", "refund-demo", "--instance", "nope")
        assert result.exit_code == 1
        assert "wf_refun" in result.output


def test_the_renderer_is_found_under_either_name():
    """boundflow/boundflow#93 renames `boundflow.cli._output` to `.output`. It is a
    rename and a docstring — the same three functions — and until it lands in a
    release, the published branch has the old name. Charter has to run against what
    is actually installable, so it accepts both.

    Delete this, and the fallback import, once #93 is in a release we can pin.
    """
    import importlib

    found = []
    for name in ("boundflow.cli.output", "boundflow.cli._output"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        found.append(name)
        assert all(callable(getattr(mod, f, None))
                   for f in ("output", "set_json", "is_json")), name
    assert found, "neither renderer module is importable"
