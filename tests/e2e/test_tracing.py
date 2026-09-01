"""Traces, end to end: a real control plane, a real worker, a file on disk.

`trace_sink` parsed and validated for as long as it existed while `charter worker`
never passed it to BoundFlow, so every worker ran with tracing off however the
manifest read. Unit tests cover the wiring; this is the one that would have caught
it, because it asserts the file the operator was promised.
"""
from __future__ import annotations

import json
import shutil
import sys

import pytest
import yaml

from charter.config.loader import load_project
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_run
from tests.e2e.harness import factory, scripted, submits
from tests.e2e.test_lifecycle import PLAYGROUND, one_instance

pytestmark = pytest.mark.asyncio


@pytest.fixture
def traced_project(tmp_path, tenant, store_url):
    """The playground, writing traces beside itself."""
    dst = tmp_path / "playground"
    shutil.copytree(PLAYGROUND, dst)
    raw = yaml.safe_load((dst / "worker.yaml").read_text())
    raw["control_plane"]["tenant"] = tenant.name
    raw["store"] = {"url": store_url}
    raw.pop("notifications", None)
    raw["trace_sink"] = {"kind": "jsonl", "path": str(tmp_path / "traces.jsonl")}
    (dst / "worker.yaml").write_text(yaml.safe_dump(raw))
    for agent in ("refund-demo", "ticket-sweeper", "delegator"):
        v1 = dst / agent / "v1.yaml"
        if v1.exists():
            v1.write_text(v1.read_text()
                          .replace('command: python', f'command: {sys.executable}')
                          .replace('args: ["mcp_server.py"]',
                                   f'args: ["{PLAYGROUND / "mcp_server.py"}"]'))
    return load_project(dst / "worker.yaml"), tmp_path / "traces.jsonl"


async def test_a_task_writes_a_trace_where_the_manifest_says(
        cp, traced_project, tenant):
    """The whole chain: worker.yaml names a sink, the worker hands it to BoundFlow,
    and a completed task leaves a trace naming the agent that ran."""
    project, traces = traced_project
    wf = await one_instance(cp, project, "ticket-sweeper", tenant)

    worker = CharterWorker(project, chat_model=factory(scripted(submits())))
    async with running(worker):
        await wait_for_run(
            cp, await cp.invoke_workflow(wf.id, context={}), timeout=90)
    await worker.aclose()

    assert traces.exists(), "the manifest asked for traces and none were written"
    written = [json.loads(line) for line in traces.read_text().splitlines() if line]
    assert any(t["workflow_type"] == "ticket-sweeper" for t in written), written
