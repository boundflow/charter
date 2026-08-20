"""The harness capabilities Charter configures, actually exercised.

Everything here was previously config validated at load time with nothing proving
it reached the agent. A rule that parses and never fires is worse than no rule,
and the only way to tell the difference is to make the agent try.

A refusal is just a message coming back — the task completes either way — so the
assertions read what the model was *told*, via the scripted model's `received`.
That is the same surface an operator would have to reason about.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

from charter.config.loader import load_project
from charter.worker import CharterWorker
from tests.e2e.conftest import running, wait_for_gate, wait_for_run
from tests.e2e.harness import calls, factory, scripted, submits, texts
from tests.e2e.test_lifecycle import one_instance

pytestmark = pytest.mark.asyncio

PLAYGROUND = Path(__file__).parent.parent.parent / "playground"
AGENT = "ticket-sweeper"


@pytest.fixture
def project(tmp_path, tenant, store_url):
    """A copy of the playground each test can edit before loading."""
    dst = tmp_path / "playground"
    shutil.copytree(PLAYGROUND, dst)
    raw = yaml.safe_load((dst / "worker.yaml").read_text())
    raw["control_plane"]["tenant"] = tenant.name
    raw["store"] = {"url": store_url}
    raw.pop("notifications", None)
    (dst / "worker.yaml").write_text(yaml.safe_dump(raw))
    for agent in ("refund-demo", "ticket-sweeper", "delegator"):
        v1 = dst / agent / "v1.yaml"
        if v1.exists():
            v1.write_text(v1.read_text()
                          .replace("command: python", f"command: {sys.executable}")
                          .replace('args: ["mcp_server.py"]',
                                   f'args: ["{PLAYGROUND / "mcp_server.py"}"]'))
    return dst


def configure(root: Path, agent: str = AGENT, **fields):
    """Edit an agent's versioned config, then reload the project."""
    path = root / agent / "v1.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.update(fields)
    path.write_text(yaml.safe_dump(raw))
    return load_project(root / "worker.yaml")


def runtime(root: Path, agent: str = AGENT, **per_run):
    path = root / agent / "runtime.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["per_run"].update(per_run)
    path.write_text(yaml.safe_dump(raw))
    return load_project(root / "worker.yaml")


async def run_with(cp, loaded, tenant, model, agent: str = AGENT, **context):
    wf = await one_instance(cp, loaded, agent, tenant)
    worker = CharterWorker(loaded, chat_model=factory(model))
    async with running(worker):
        info = await wait_for_run(
            cp, await cp.invoke_workflow(wf.id, context=context), timeout=90)
    await worker.aclose()
    return info


# ── file rules ───────────────────────────────────────────────────────────────


async def test_a_deny_rule_refuses_the_write(cp, project, tenant):
    """The rule is enforced by the harness from policy we translated, so this is
    the whole chain: our yaml, their permission check, the agent's error."""
    loaded = configure(project, file_rules=[
        {"operations": ["write"], "paths": ["/secrets/**"], "mode": "deny"}])
    model = scripted(
        calls("write_file", file_path="/secrets/keys.txt", content="x"),
        submits(summary="could not write", needs_attention=0),
    )

    await run_with(cp, loaded, tenant, model)

    assert "permission denied" in texts(model.received).lower()


async def test_an_allowed_path_is_written(cp, project, tenant):
    """The other half. A rule that refused everything would pass the test above
    while making the agent useless."""
    loaded = configure(project, file_rules=[
        {"operations": ["write"], "paths": ["/secrets/**"], "mode": "deny"}])
    model = scripted(
        calls("write_file", file_path="/work/notes.md", content="hello"),
        calls("read_file", file_path="/work/notes.md"),
        submits(summary="wrote and read it back", needs_attention=0),
    )

    await run_with(cp, loaded, tenant, model)

    told = texts(model.received)
    assert "permission denied" not in told.lower()
    assert "hello" in told


# ── the capability allowlist ─────────────────────────────────────────────────


async def test_a_capability_outside_the_allowlist_is_refused(cp, project, tenant):
    """Default-deny over the harness's own tools. Nothing upstream does this —
    `permissions` covers filesystem paths, not which capabilities exist at all."""
    loaded = configure(project, allowed_capabilities=["read"])
    model = scripted(
        calls("write_file", file_path="/work/notes.md", content="x"),
        submits(summary="could not write", needs_attention=0),
    )

    await run_with(cp, loaded, tenant, model)

    told = texts(model.received).lower()
    assert "write_file" in told
    assert "not" in told or "refus" in told or "denied" in told


async def test_an_allowed_capability_still_works(cp, project, tenant):
    loaded = configure(project, allowed_capabilities=["read", "write"])
    model = scripted(
        calls("write_file", file_path="/work/notes.md", content="allowed"),
        calls("read_file", file_path="/work/notes.md"),
        submits(summary="fine", needs_attention=0),
    )

    await run_with(cp, loaded, tenant, model)
    assert "allowed" in texts(model.received)


# ── capability call limits ───────────────────────────────────────────────────


async def test_a_capability_cap_survives_the_agent_switching_tools(cp, project, tenant):
    """The reason capabilities exist rather than per-tool caps: cap `write_file`
    and the agent reaches for `edit_file`. `write` covers both."""
    loaded = runtime(project, capability_call_limits=[
        {"capability": "write", "max_calls": 1}])
    model = scripted(
        calls("write_file", file_path="/work/a.md", content="first"),
        calls("edit_file", file_path="/work/a.md", old_string="first",
              new_string="second"),
        submits(summary="one write got through", needs_attention=0),
    )

    await run_with(cp, loaded, tenant, model)

    told = texts(model.received).lower()
    assert "limit" in told or "cap" in told


# ── skills ───────────────────────────────────────────────────────────────────


async def test_a_skill_reaches_the_prompt(cp, project, tenant):
    """Charter ships `v<N>/skills/` and the harness's own loader reads it. Nothing
    verified the second half — a directory on disk that no agent ever sees is the
    failure this catches."""
    skill = project / AGENT / "v1" / "skills" / "house-style"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        "---\nname: house-style\ndescription: How we word things.\n---\n\n"
        "Always mention the ticket id in the first sentence.")
    loaded = load_project(project / "worker.yaml")

    model = scripted(submits(summary="done", needs_attention=0))
    await run_with(cp, loaded, tenant, model)

    assert "house-style" in texts(model.received)


# ── asking a human ───────────────────────────────────────────────────────────


@pytest.mark.xfail(reason="the ask tool parks correctly but the run doesn't resume "
                          "after submit_input — only one model call is ever made. "
                          "Unproven end to end; the unit tests cover the branch.",
                   strict=True)
async def test_the_agent_can_ask_and_the_answer_reaches_it(cp, project, tenant):
    """The harness only ever stops at a tool call, so asking is a tool. This is the
    whole path: the tool exists, calling it parks on AwaitInput, and the answer
    comes back as the tool's result."""
    loaded = configure(project, ask_human={"when": "eagerly", "timeout_seconds": 120})
    model = scripted(
        calls("ask_human", question="Which ticket should I start with?"),
        submits(summary="started with 4821 as instructed", needs_attention=1),
    )

    wf = await one_instance(cp, loaded, AGENT, tenant)
    worker = CharterWorker(loaded, chat_model=factory(model))
    async with running(worker):
        request_id = await cp.invoke_workflow(wf.id)
        info = await _wait_for_input(cp, wf.id)
        # An answer is a dict, not a string — `ctx.input_answer` hands the loop
        # structure and the harness's `respond` wants a message.
        await cp.submit_input(wf.id, info.input_id, {"answer": "start with 4821"},
                              "e2e")
        run = await wait_for_run(cp, request_id, timeout=90)
    await worker.aclose()

    assert run.status.value == "completed", run.failure_reason
    assert "4821" in texts(model.received)


async def _wait_for_input(cp, workflow_id, timeout: int = 90):
    import asyncio

    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        wf = await cp.get_workflow(workflow_id)
        if wf.pending_input:
            return wf.pending_input
        assert asyncio.get_event_loop().time() < deadline, "no question was asked"
        await asyncio.sleep(0.3)


# ── subagents ────────────────────────────────────────────────────────────────


async def test_a_subagent_runs_and_spends_the_parents_budget(cp, project, tenant):
    """Sync subagents share the parent's governed model, so their calls land in the
    same pool. That is what makes the concurrency fix in begin_call load-bearing."""
    loaded = runtime(project, max_total_subagents=3)
    model = scripted(
        calls("task", description="summarise ticket 4821",
              subagent_type="general-purpose"),
        submits(summary="delegated", needs_attention=1),
    )

    info = await run_with(cp, loaded, tenant, model)
    assert info.status.value == "completed", info.failure_reason


async def test_a_spent_subagent_cap_is_refused_and_the_agent_carries_on(cp, project, tenant):
    """Refused rather than raised: a limit should degrade the work, not end it."""
    loaded = runtime(project, max_total_subagents=1)
    model = scripted(
        calls("task", description="first", subagent_type="general-purpose"),
        calls("task", description="second", subagent_type="general-purpose"),
        submits(summary="did the rest myself", needs_attention=1),
    )

    info = await run_with(cp, loaded, tenant, model)

    assert info.status.value == "completed", info.failure_reason
    assert "subagent limit reached" in texts(model.received).lower()
