"""`--inputs-file`, end to end: the JSON a task arrives in must reach the model.

The CLI parsed `--inputs-file /tmp/task.json` as an input *named* `inputs_file`
whose value was the path, so the worker saw every declared input as missing and
failed the task before the model was called — `llm_calls: 0`, failure reason
"ticket_id was not given, and the objective needs it". Flags worked, which is
what pinned it to the CLI rather than the loop.

Both spellings are run here — a file, and `-` for stdin — through the real `run`
command, against the real control plane, with a scripted model. The model records
everything it is handed, so the objective actually containing the value is the
thing asserted, not just a completed task.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from charter.config.loader import load_project
from charter.provisioning.apply import apply_project, compile_agent, create_instance
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_run
from tests.e2e.harness import factory, scripted, submits, texts

pytestmark = pytest.mark.asyncio

# The repro agent: one required input the objective renders, one result field.
# No tools, no skills — nothing here but the input path under test.
AGENT = """
apiVersion: charter/v1
kind: AgentConfig
name: input-file-repro
version: 1
model: claude-haiku-4-5
objective: |
  Report the ticket id {{ inputs.ticket_id }}.
inputs:
  ticket_id: { type: string, required: true }
response_format:
  answer:
    type: string
    description: The ticket id.
"""

WORKER = """
apiVersion: charter/v1
kind: Worker
name: input-file-repro
control_plane:
  endpoint: {endpoint}
  worker_endpoint: {worker_endpoint}
  api_key: {api_key}
  tenant: {tenant}
store:
  url: {store}
agents_dir: ./
llm:
  provider: anthropic
  api_key: unused-the-harness-scripts-the-model
serves:
- agent: input-file-repro
  versions: [1]
model_pricing:
  claude-haiku-4-5:
    input: 0.8
    output: 4.0
"""

# The request id is echoed by `ui.detail`, which goes to stderr when stdout
# isn't a terminal — a subprocess is never a terminal. ANSI-stripped, because
# typer styles ids bold.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STATUS = re.compile(r"charter status (\S+)")
REPO = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def project(tmp_path, tenant, store_url, monkeypatch):
    """A one-agent project, resolving by value what worker.yaml would take from
    the environment — the endpoint and key are the session's, not a file's."""
    def _(name):
        return os.environ[name]

    d = tmp_path / "repro"
    (d / "input-file-repro").mkdir(parents=True)
    (d / "input-file-repro" / "v1.yaml").write_text(AGENT.lstrip())
    (d / "worker.yaml").write_text(WORKER.format(
        endpoint=_("BOUNDFLOW_SERVER_ADDRESS"),
        worker_endpoint=_("BOUNDFLOW_WORKER_ADDRESS"),
        api_key=_("BOUNDFLOW_API_KEY"),
        tenant=tenant.name, store=store_url))
    # The CLI builds its own client from the environment, not from worker.yaml.
    monkeypatch.setenv("CHARTER_TENANT", tenant.name)
    return load_project(d / "worker.yaml")


async def _instance(cp, project, tenant):
    bundle = project.agents["input-file-repro"]
    wf = await create_instance(cp, compile_agent(bundle), tenant.id)
    await apply_project(cp, project, only="input-file-repro", all_=True)
    return wf


def _cli(*args, stdin=None):
    """The command a person types, in its own process. `run` ends with
    `asyncio.run`, which cannot share a loop with this test's worker — and
    wrapping it in a thread breaks the worker's dispatch stream mid-task — so
    the process boundary is what lets the CLI and the worker coexist."""
    return subprocess.run(
        [sys.executable, "-c", "from charter.cli import main; main()", *args],
        input=stdin, capture_output=True, text=True, timeout=60,
        cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)})


async def _run_through_the_cli(cp, wf, project, tenant, *extra, stdin=None):
    """The real `run` command against the real control plane, with a scripted
    model behind the worker that picks the task up."""
    model = scripted(submits(answer="4821"))
    worker = CharterWorker(project, chat_model=factory(model))
    async with running(worker):
        res = _cli("run", "input-file-repro", "--instance", wf.id,
                   "--tenant", tenant.name, *extra, stdin=stdin)
        assert res.returncode == 0, res.stderr
        out = ANSI.sub("", res.stdout + res.stderr)
        request_id = STATUS.search(out).group(1)
        info = await wait_for_run(cp, request_id, timeout=90)
    return model, info


async def test_a_file_of_inputs_reaches_the_objective(cp, project, tenant, tmp_path):
    wf = await _instance(cp, project, tenant)
    f = tmp_path / "task.json"
    f.write_text('{"ticket_id": "4821"}')
    model, info = await _run_through_the_cli(
        cp, wf, project, tenant, "--inputs-file", str(f))

    assert info.status.value == "completed", info.failure_reason
    # The model was handed the objective with the value rendered in — the input
    # travelled from the file, through the invoke context, into the prompt.
    assert "Report the ticket id 4821." in texts(model.received)
    assert info.result == {"answer": "4821"}
    # In the bug the task failed before any call; here the one call it took.
    assert (await cp.get_workflow_metrics(wf.id)).total_llm_calls == 1


async def test_stdin_dash_reaches_the_objective(cp, project, tenant):
    wf = await _instance(cp, project, tenant)
    model, info = await _run_through_the_cli(
        cp, wf, project, tenant, "--inputs-file", "-",
        stdin='{"ticket_id": "4821"}')

    assert info.status.value == "completed", info.failure_reason
    assert "Report the ticket id 4821." in texts(model.received)
    assert info.result == {"answer": "4821"}
    assert (await cp.get_workflow_metrics(wf.id)).total_llm_calls == 1


async def test_the_flag_form_still_reaches_the_objective(cp, project, tenant):
    """The path that always worked, run the same way — a regression here is what
    the two fixes above would be mistaken for."""
    wf = await _instance(cp, project, tenant)
    model, info = await _run_through_the_cli(
        cp, wf, project, tenant, "--ticket_id", "4821")

    assert info.status.value == "completed", info.failure_reason
    assert "Report the ticket id 4821." in texts(model.received)
    assert info.result == {"answer": "4821"}
    assert (await cp.get_workflow_metrics(wf.id)).total_llm_calls == 1
