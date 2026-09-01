"""The trace sink a worker builds from its manifest.

`trace_sink` validated and did nothing for as long as it existed: the manifest
parsed it, `BoundFlowWorker` accepted one, and `charter worker` never passed it. So
these cover the wiring as much as the sinks.
"""
import ast
import inspect
import json
from pathlib import Path

import pytest

from charter import trace
from charter.config.worker import TraceSink


def _trace(**kw):
    from boundflow.trace import OperationTrace
    return OperationTrace(**{
        "trace_id": "req_1", "workflow_id": "wf_1", "workflow_type": "leads-finder",
        "version": 1, "operation": "invoke_entry", "outcome": "completed",
        "failed": False, "start_ms": 0, "end_ms": 10, "agent_runs": [], **kw})


class TestBuild:
    def test_no_sink_declared_is_no_sink(self):
        assert trace.build(None) is None

    def test_kind_none_is_no_sink(self):
        """Declaring it off has to reach the worker as off, not as a sink that
        drops what it is given."""
        assert trace.build(TraceSink(kind="none")) is None

    def test_logging_and_jsonl(self, tmp_path):
        from boundflow.trace import JsonlFileTraceSink, LoggingTraceSink

        assert isinstance(trace.build(TraceSink(kind="logging")), LoggingTraceSink)
        built = trace.build(TraceSink(kind="jsonl", path=str(tmp_path / "t.jsonl")))
        assert isinstance(built, JsonlFileTraceSink)

    def test_otel_without_its_extra_says_which_one(self):
        """The dependency is optional, so the failure has to name the install rather
        than surface opentelemetry's own ImportError."""
        pytest.importorskip  # noqa: B018 - documents that this asserts the absence
        try:
            import opentelemetry.sdk  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match=r"charter\[otel\]"):
                trace.build(TraceSink(kind="otel", endpoint="http://localhost:4317"))

    def test_a_path_resolves_from_the_environment(self, tmp_path, monkeypatch):
        """Every other credential and path in worker.yaml is a ${VAR}; a trace path
        written to a container's read-only mount is the failure this prevents."""
        monkeypatch.setenv("CHARTER_TRACE_PATH", str(tmp_path / "from-env.jsonl"))
        built = trace.build(TraceSink(kind="jsonl", path="${CHARTER_TRACE_PATH}"))
        assert str(tmp_path / "from-env.jsonl") in repr(vars(built))


class TestTheSinkActuallyWrites:
    @pytest.mark.asyncio
    async def test_a_trace_lands_in_the_file(self, tmp_path):
        """End of the line: a real OperationTrace through the sink the manifest
        asked for, read back off disk."""
        path = tmp_path / "traces.jsonl"
        sink = trace.build(TraceSink(kind="jsonl", path=str(path)))

        await sink.emit(_trace())

        written = json.loads(path.read_text().strip())
        assert written["trace_id"] == "req_1"
        assert written["workflow_type"] == "leads-finder"


def test_the_worker_hands_its_sink_to_boundflow():
    """The bug this file exists for. `trace_sink` parsed, validated and was never
    passed, so every worker ran with tracing off however the manifest read."""
    import charter.worker as worker

    tree = ast.parse(Path(inspect.getfile(worker)).read_text())
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "BoundFlowWorker")
    assert "trace_sink" in {kw.arg for kw in call.keywords}


def test_a_manifest_on_disk_reaches_the_worker_it_configures(tmp_path):
    """The whole chain offline: worker.yaml -> load_project -> build -> the worker
    BoundFlow will run. Only the control plane dispatching a run is missing, and
    that lives in tests/e2e."""
    import shutil

    import yaml
    from boundflow import BoundFlowWorker
    from boundflow.trace import JsonlFileTraceSink

    from charter.config.loader import load_project

    project = tmp_path / "project"
    shutil.copytree(Path(__file__).parent.parent / "examples", project)
    manifest_path = project / "worker.yaml"
    raw = yaml.safe_load(manifest_path.read_text())
    raw["trace_sink"] = {"kind": "jsonl", "path": str(tmp_path / "traces.jsonl")}
    manifest_path.write_text(yaml.safe_dump(raw))

    manifest = load_project(manifest_path).manifest
    worker = BoundFlowWorker(llm=object(), api_key="k",
                             trace_sink=trace.build(manifest.trace_sink,
                                                    manifest.name or "charter"))

    # Private, because the alternative is a control plane dispatching a real run.
    assert isinstance(worker._trace_sink, JsonlFileTraceSink)
