"""What the worker is handed from its manifest.

BoundFlow serves the control API and worker dispatch on different addresses. Only
`endpoint` was ever read, so a worker.yaml pointing at a remote control plane gave
a CLI talking to it and a worker dialling localhost.
"""
import ast
import asyncio
import inspect
from pathlib import Path

import pytest

from charter.config.worker import ControlPlane


def _worker_call():
    import charter.worker as worker

    tree = ast.parse(Path(inspect.getfile(worker)).read_text())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "BoundFlowWorker")


def test_the_worker_is_told_where_to_claim_tasks():
    """Unset, BoundFlow falls back to localhost — right while developing, and wrong
    every other time."""
    assert "address" in {kw.arg for kw in _worker_call().keywords}


def test_the_worker_is_told_where_to_send_traces():
    """`trace_sink` parsed and validated for as long as it existed while nothing
    passed it, so tracing was off however worker.yaml read."""
    assert "trace_sink" in {kw.arg for kw in _worker_call().keywords}


def test_a_worker_endpoint_is_optional():
    """Local development has both addresses on localhost, so requiring it would
    break every existing project to fix a case only a remote control plane has."""
    cp = ControlPlane(endpoint="http://localhost:50051", api_key="k", tenant="t")
    assert cp.worker_endpoint == ""


def test_a_worker_endpoint_is_carried_when_given():
    cp = ControlPlane(endpoint="https://api.example:443", api_key="k", tenant="t",
                      worker_endpoint="https://worker.example:443")
    assert cp.worker_endpoint == "https://worker.example:443"


def test_a_worker_with_an_unresolved_model_key_refuses_to_start(tmp_path, monkeypatch):
    """An unset ${VAR} in `llm` stops the worker before it claims anything.

    Resolved lazily, the worker boots, prints every agent as ready, and dies
    mid-dispatch on the first real task — which fails that task and reads as a
    Charter bug rather than a missing export.
    """
    from charter import scaffold
    from charter.config.loader import load_project
    from charter.worker import run_worker

    for rel, body in scaffold.files("triage").items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def built(*a, **k):
        raise AssertionError("the worker was constructed before the key was checked")

    monkeypatch.setattr("charter.worker.CharterWorker", built)

    with pytest.raises(RuntimeError) as e:
        asyncio.run(run_worker(load_project(tmp_path / "worker.yaml")))
    assert "ANTHROPIC_API_KEY" in str(e.value)


def test_a_checkout_worker_takes_its_limits_from_the_control_plane(monkeypatch):
    """runtime.yaml is policy, so `charter apply` owns it and the worker reads it
    back. Read off local disk instead, a lowered cap needs a worker restart, and
    two workers serving the same agent from different checkouts enforce different
    numbers. The registry path always did this; the checkout path did not.
    """
    from types import SimpleNamespace

    from charter.config.loader import load_agent
    from charter.config.worker import Served as ServedSpec
    from charter.worker import CharterWorker

    bundle = load_agent(Path(__file__).parent.parent / "examples" / "refund-triage")
    assert bundle.runtime.per_run.max_cost_usd == 0.30, "the file on disk"

    applied = SimpleNamespace(custom={}, max_cost_usd=0.05, max_llm_calls=7)
    cp = SimpleNamespace(
        get_agent_runtime_policy=lambda *_: _async(applied),
        list_workflows=lambda: _async([]))

    stub = SimpleNamespace(
        project=SimpleNamespace(agents={"refund-triage": bundle}),
        _tenant_id="t",
        _workflow_for=lambda *_: _async(SimpleNamespace(id="w")))

    spec = ServedSpec(agent="refund-triage", versions=[1])
    got, _ = asyncio.run(CharterWorker._bundle_for(stub, cp, spec))

    assert got.runtime.per_run.max_cost_usd == 0.05, "applied policy, not the file"
    assert got.runtime.per_run.max_llm_calls == 7


async def _async(value):
    return value
