"""What the worker is handed from its manifest.

BoundFlow serves the control API and worker dispatch on different addresses. Only
`endpoint` was ever read, so a worker.yaml pointing at a remote control plane gave
a CLI talking to it and a worker dialling localhost.
"""
import ast
import inspect
from pathlib import Path

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
