"""End-to-end fixtures: a real control plane, a real MCP subprocess, a fake model.

Everything except the model is real. That's the cut that matters — every bug this
project has hit came from a boundary a fake didn't model:

  * dotted tool names 400'd, because only a real provider enforces the charset
  * apply wasn't idempotent, because the fake control plane accepted activate
    unconditionally while the server requires paused/cooldown
  * resume didn't pass the policy-decision id, for the same reason
  * MCP's is_error was read under its 1.x name, and only a real server caught it

The model is faked because it's the one component where determinism is worth more
than fidelity — and a scripted chat model still records what it was *offered*, so
what a config promises about the model's authority stays checkable from outside.
See `harness.py`.

Postgres is real too now, and not incidentally: the harness keeps its conversation
and its files there, so a task resuming after a gate is only testable against a
real store.

    docker compose -f ../convergeplane/docker-compose.dist.yml up -d
    export BOUNDFLOW_API_KEY=... BOUNDFLOW_SERVER_ADDRESS=http://localhost:50051
    export CHARTER_STORE_URL=postgres://boundflow:boundflow@localhost:5433/boundflow
    pytest tests/e2e
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from boundflow import ControlPlaneClient

SERVER_ADDRESS = os.environ.get("BOUNDFLOW_SERVER_ADDRESS", "http://localhost:50051")


@pytest.fixture(scope="session")
def store_url():
    """Where the harness keeps state. Without it there is nothing to resume from,
    so these tests would be checking a different system."""
    url = os.environ.get("CHARTER_STORE_URL")
    if not url:
        pytest.skip("CHARTER_STORE_URL not set — the harness needs a store")
    return url


@pytest.fixture(scope="session")
def boundflow_api_key():
    key = os.environ.get("BOUNDFLOW_API_KEY")
    if not key:
        pytest.skip("BOUNDFLOW_API_KEY not set — start the control plane first")
    return key


@pytest_asyncio.fixture
async def cp(boundflow_api_key):
    async with ControlPlaneClient(SERVER_ADDRESS, api_key=boundflow_api_key) as client:
        yield client


@pytest_asyncio.fixture
async def tenant(cp):
    """A fresh tenant per test. Agent identity is (tenant, name), so isolation is
    what lets tests reuse agent names without colliding."""
    return await cp.create_tenant(f"charter-e2e-{uuid.uuid4().hex[:8]}")


async def wait_for_run(cp, request_id: str, timeout: int = 60):
    """Poll one run until terminal. Keyed to the run rather than the workflow's
    aggregate state, which would false-positive during the pre-scheduled window."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        info = await cp.get_request_info(request_id)
        if info.status.value in ("completed", "failed"):
            return info
        assert asyncio.get_event_loop().time() < deadline, (
            f"run {request_id} never finished (last: {info.status.value})")
        await asyncio.sleep(0.3)


async def wait_for_gate(cp, workflow_id: str, timeout: int = 60):
    """Wait until a workflow is parked, and return the open approval gate."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        wf = await cp.get_workflow(workflow_id)
        if wf.pending_approval:
            return wf.pending_approval
        assert asyncio.get_event_loop().time() < deadline, (
            f"no gate opened (state: {wf.lifecycle_state.value})")
        await asyncio.sleep(0.3)


@asynccontextmanager
async def running(worker):
    task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.2)
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # Cancelling the task doesn't close the worker: its control-plane session
        # and its MCP subprocesses outlive it. A lingering worker keeps polling and
        # can claim the *next* test's job — running it against the wrong scripted
        # model, which is why these pass alone and fail together.
        with contextlib.suppress(Exception):
            await worker.aclose()
