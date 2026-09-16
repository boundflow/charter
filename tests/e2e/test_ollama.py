"""The quickstart against a model running on this machine.

Nothing here is faked, the model least of all — which is the point. A local
runtime is not a provider with a different base URL: its client rejects call
arguments the hosted ones take, and every fake in this suite accepts any argument,
so that class of bug is invisible everywhere else. Two of them shipped. Both broke
every Ollama run, and both are fixed in the boundflow the pin floors at:

  * the output cap was sent as `max_tokens`, which Ollama's client has no
    parameter for
  * a spent `max_llm_calls` forced the ending tool with `tool_choice`, which it
    also has no parameter for, so the run crashed instead of finishing

So these assert on structure and acceptance: that the provider Charter builds from
`worker.yaml` takes the request, that the injected schema comes back filled, and
that a run out of budget ends rather than raises. What the model says is not under
test.

The agent is `charter init`'s own scaffold, so this is the README's quickstart with
one line changed.

    ollama pull qwen2.5:7b
    OLLAMA_TEST_MODEL=qwen2.5:7b pytest tests/e2e/test_ollama.py

Not in CI, and deliberately: a model small enough to pull on every run is too small
to drive the harness — `qwen2.5:3b` answered by writing a file instead of calling
`submit_result`. The SDK-level contract with Ollama is covered by BoundFlow's own
`Python SDK (Ollama)` job, which can use a 0.5B model because it asserts on the
call, not on the agent. This asserts on the agent, so it wants a real one. They
skip without `OLLAMA_TEST_MODEL`, which means a run without it looks identical to a
passing one.
"""
from __future__ import annotations

import os

import pytest
import yaml

from charter import scaffold
from charter.config.loader import load_project
from charter.provisioning.apply import apply_project, compile_agent, create_instance
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_run

# Inference on a CPU takes minutes where a hosted call takes seconds, and every
# timeout below is sized for that rather than for the defaults — including the
# suite's own, which is shorter than a single local call.
CALL_SECONDS = 600
RUN_TIMEOUT = 900

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.timeout(RUN_TIMEOUT + 120),
    pytest.mark.skipif(not os.environ.get("OLLAMA_TEST_MODEL"),
                       reason="OLLAMA_TEST_MODEL not set"),
]


def quickstart(tmp_path, tenant, store_url, *, max_llm_calls: int):
    """`charter init triage`, pointed at a local model and this test's tenant."""
    root = tmp_path / "local"
    root.mkdir()
    for name, text in scaffold.files("triage").items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    v1 = root / "triage" / "v1.yaml"
    v1.write_text(v1.read_text().replace("model: claude-haiku-4-5",
                                         f"model: {os.environ['OLLAMA_TEST_MODEL']}"))

    worker = root / "worker.yaml"
    raw = yaml.safe_load(worker.read_text())
    # No api_key: a local runtime needs none, and this is the provider build under
    # test — the model is the one Charter constructs, not one the test hands it.
    raw["llm"] = {"provider": "ollama"}
    raw["control_plane"]["tenant"] = tenant.name
    raw["store"] = {"url": store_url}
    worker.write_text(yaml.safe_dump(raw))

    # An unpriced model reports no cost, so a dollar budget never binds and calls
    # are the only ceiling that does.
    (root / "triage" / "runtime.yaml").write_text(yaml.safe_dump({
        "apiVersion": "charter/v1", "kind": "RuntimePolicy", "agent": "triage",
        "per_run": {"max_llm_calls": max_llm_calls},
        "limits": {"max_call_seconds": CALL_SECONDS},
    }))
    return load_project(worker)


async def run_one(cp, project, tenant, ticket: str):
    bundle = project.agents["triage"]
    wf = await create_instance(cp, compile_agent(bundle), tenant.id)
    await apply_project(cp, project, only="triage", all_=True)

    worker = CharterWorker(project)
    async with running(worker):
        request_id = await cp.invoke_workflow(wf.id, context={"ticket": ticket})
        info = await wait_for_run(cp, request_id, timeout=RUN_TIMEOUT)
    await worker.aclose()
    return info


async def test_a_local_model_completes_a_task(cp, tenant, store_url, tmp_path):
    """The whole chain on a model with no key and no vendor: Charter builds the
    provider from `worker.yaml`, the cap goes on the field that provider declares,
    and the injected result schema comes back filled."""
    project = quickstart(tmp_path, tenant, store_url, max_llm_calls=4)
    info = await run_one(cp, project, tenant,
                         "card declined twice, tried a new one")

    assert info.run_outcome.value == "successful", info.failure_reason
    for field in ("category", "next_step"):
        assert isinstance(info.result[field], str) and info.result[field].strip()


async def test_a_run_out_of_calls_ends_on_the_finalizer(cp, tenant, store_url,
                                                        tmp_path):
    """One permitted call is also the last one, so the first thing the model is
    asked to do is the forced ending — the path that used to crash the run.

    Ollama cannot force a tool at all, so the ending holds because the finalizer is
    the only tool left on offer, not because the provider compelled it.
    """
    project = quickstart(tmp_path, tenant, store_url, max_llm_calls=1)
    info = await run_one(cp, project, tenant, "refund my last order")

    assert info.run_outcome.value == "successful", info.failure_reason
    assert isinstance(info.result["category"], str)
