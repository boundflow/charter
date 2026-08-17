"""The `charter` CLI.

A sibling of the `boundflow` CLI, not a layer on top: both are Typer apps calling
ControlPlaneClient over gRPC. Charter never shells out to `boundflow` — that would
turn validated models into flag strings and typed exceptions into exit codes.

Only `validate` and `apply` live here so far; the rest of the surface (run, status,
approve, pending, memory, worker) lands with the worker.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

import typer

from .compile import compile_agent
from .config.loader import ConfigError, load_agent, load_project

app = typer.Typer(add_completion=False, help="Declarative, governed agents on BoundFlow.")

ENV_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _err(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)


def _warn(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.YELLOW)


def _ok(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.GREEN)


def _resolve(value: str) -> str:
    """Expand ${VAR} references from the environment. Secrets live in the
    environment, never in a file — worker.yaml is committed too."""
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in os.environ:
            missing.append(name)
            return ""
        return os.environ[name]

    out = ENV_REF.sub(sub, value)
    if missing:
        raise ConfigError([f"environment variable {n} is not set" for n in missing])
    return out


def _load(path: Path):
    """Load whatever `path` points at — a worker manifest, a project directory, or a
    single agent directory."""
    path = Path(path)
    if path.is_file():
        return load_project(path)
    if (path / "worker.yaml").exists():
        return load_project(path / "worker.yaml")
    return load_agent(path)


@app.command()
def validate(
    path: Path = typer.Argument(Path("."), help="worker.yaml, a project dir, or an agent dir"),
) -> None:
    """Parse and cross-check every file. Touches no network."""
    try:
        loaded = _load(path)
    except ConfigError as e:
        _err(f"{len(e.problems)} problem(s):")
        for p in e.problems:
            _err(f"  - {p}")
        raise typer.Exit(1)

    if hasattr(loaded, "agents"):
        for name, bundle in sorted(loaded.agents.items()):
            versions = ", ".join(f"v{v}" for v in sorted(bundle.versions))
            typer.echo(f"  {name}  {versions}  ({bundle.latest.invoke_mode})")
        _ok(f"ok — {len(loaded.agents)} agent(s)")
    else:
        versions = ", ".join(f"v{v}" for v in sorted(loaded.versions))
        typer.echo(f"  {loaded.name}  {versions}  ({loaded.latest.invoke_mode})")
        _ok("ok")


@app.command()
def apply(
    path: Path = typer.Argument(Path("."), help="worker.yaml or a project dir"),
    only: str = typer.Option(None, "--agent", help="Apply just this agent"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print, don't call"),
) -> None:
    """Validate, then create or update workflows, policies, and pricing. Idempotent."""
    try:
        project = _load(path)
    except ConfigError as e:
        _err(f"{len(e.problems)} problem(s):")
        for p in e.problems:
            _err(f"  - {p}")
        raise typer.Exit(1)

    # A bare agent directory applies straight from the environment — the smallest
    # path is one v1.yaml plus BOUNDFLOW_*. A worker manifest adds fleet concerns
    # (which versions are served, where approvals go, pricing), not permission.
    if not hasattr(project, "manifest"):
        _apply_single(project, dry_run=dry_run)
        return

    if dry_run:
        for served in project.manifest.serves:
            if only and served.agent != only:
                continue
            bundle = project.agents[served.agent]
            version = max(v for v in served.versions if v in bundle.versions)
            c = compile_agent(bundle, version)
            typer.echo(f"\n{c.name} v{c.version}  ({c.workflow_config.invoke_mode.value})")
            typer.echo(f"  runtime   {c.runtime_policy.model_dump(exclude_defaults=True)}")
            for rule in c.workflow_rules:
                typer.echo(f"  lifecycle {rule.metric.value} >= {rule.threshold} -> "
                           f"{rule.action.model_dump()}")
        _ok("\ndry run — nothing applied")
        return

    from boundflow import ControlPlaneClient

    from .provisioning.apply import apply_project

    cp_cfg = project.manifest.control_plane
    try:
        endpoint, api_key = _resolve(cp_cfg.endpoint), _resolve(cp_cfg.api_key)
        # Resolve the tenant too, so a missing var fails before any call is made.
        project.manifest.control_plane.tenant_id = _resolve(cp_cfg.tenant_id)
    except ConfigError as e:
        for p in e.problems:
            _err(f"  - {p}")
        raise typer.Exit(1)

    async def run() -> None:
        async with ControlPlaneClient(endpoint, api_key) as cp:
            for r in await apply_project(cp, project, only=only):
                verb = "created" if r.created else "updated"
                _ok(f"{verb}  {r.agent} v{r.version}  ({r.workflow_id})")
                for w in r.warnings:
                    _warn(f"  warning: {w}")

    try:
        asyncio.run(run())
    except Exception as e:  # noqa: BLE001 — the CLI is the boundary; show it plainly
        _err(f"apply failed: {type(e).__name__}: {e}")
        raise typer.Exit(1)


def _print_compiled(c) -> None:
    typer.echo(f"\n{c.name} v{c.version}  ({c.workflow_config.invoke_mode.value})")
    typer.echo(f"  runtime   {c.runtime_policy.model_dump(exclude_defaults=True)}")
    for rule in c.workflow_rules:
        typer.echo(f"  lifecycle {rule.metric.value} >= {rule.threshold} -> "
                   f"{rule.action.model_dump()}")


def _apply_single(bundle, *, dry_run: bool) -> None:
    """Apply one agent directory using BOUNDFLOW_* from the environment."""
    compiled = compile_agent(bundle)
    if dry_run:
        _print_compiled(compiled)
        _ok("\ndry run — nothing applied")
        return

    tenant_id = os.environ.get("BOUNDFLOW_TENANT_ID")
    if not tenant_id:
        _err("BOUNDFLOW_TENANT_ID is not set — needed to create a workflow")
        _err("  (or point charter at a worker.yaml, which carries it)")
        raise typer.Exit(1)

    from boundflow import ControlPlaneClient

    from .provisioning.apply import apply_bundle

    async def run() -> None:
        # Endpoint and key come from BOUNDFLOW_SERVER_ADDRESS / BOUNDFLOW_API_KEY.
        async with ControlPlaneClient() as cp:
            r = await apply_bundle(cp, bundle, tenant_id)
            verb = "created" if r.created else "updated"
            _ok(f"{verb}  {r.agent} v{r.version}  ({r.workflow_id})")
            for w in r.warnings:
                _warn(f"  warning: {w}")

    try:
        asyncio.run(run())
    except Exception as e:  # noqa: BLE001
        _err(f"apply failed: {type(e).__name__}: {e}")
        raise typer.Exit(1)


# ── operating an applied agent ──────────────────────────────────────────────
#
# Everything below needs only an agent NAME and BOUNDFLOW_* in the environment —
# no files. The person approving a $240 refund at 2am should not need the repo
# checked out; they got a webhook with an approval_id.


def _cp():
    from boundflow import ControlPlaneClient
    return ControlPlaneClient()


async def _workflow_for(cp, agent: str):
    for w in await cp.list_workflows():
        if w.workflow_type == agent:
            return w
    return None


def _coerce(spec, raw: str, name: str):
    """CLI flags arrive as strings; the declared type is what they must become."""
    try:
        if spec.type == "integer":
            return int(raw)
        if spec.type == "number":
            return float(raw)
        if spec.type == "boolean":
            return raw.lower() in ("1", "true", "yes", "y")
        return raw
    except ValueError:
        raise typer.BadParameter(f"--{name.replace('_', '-')} must be a {spec.type}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    agent: str = typer.Argument(..., help="Agent name (its directory)"),
    path: Path = typer.Option(Path("."), "--path", help="Where agents live"),
) -> None:
    """Start one task. Declared inputs become --flags, validated before the request
    is created so a typo fails here instead of burning a run."""
    agent_dir = Path(path) / agent
    if not agent_dir.is_dir():
        agent_dir = Path(path)
    try:
        bundle = load_agent(agent_dir)
    except ConfigError as e:
        _err(f"no agent config found for {agent!r} — `run` needs it to validate inputs")
        for p in e.problems:
            _err(f"  - {p}")
        raise typer.Exit(1)

    cfg = bundle.latest
    flags = {}
    args = list(ctx.args)
    while args:
        token = args.pop(0)
        if not token.startswith("--"):
            raise typer.BadParameter(f"unexpected argument {token!r}")
        key = token[2:].replace("-", "_")
        if "=" in key:
            key, value = key.split("=", 1)
        elif args and not args[0].startswith("--"):
            value = args.pop(0)
        else:
            value = "true"
        flags[key] = value

    unknown = set(flags) - set(cfg.inputs)
    if unknown:
        _err(f"unknown input(s): {', '.join(sorted(unknown))}")
        _err(f"  declared: {', '.join(cfg.inputs) or '(none)'}")
        raise typer.Exit(1)

    context = {}
    for name, spec in cfg.inputs.items():
        if name in flags:
            context[name] = _coerce(spec, flags[name], name)
        elif spec.default is not None:
            context[name] = spec.default
        elif spec.required:
            _err(f"--{name.replace('_', '-')} is required")
            raise typer.Exit(1)
        if spec.enum and name in context and context[name] not in spec.enum:
            _err(f"--{name.replace('_', '-')} must be one of {spec.enum}")
            raise typer.Exit(1)

    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent)
            if wf is None:
                _err(f"{agent!r} has not been applied yet — run `charter apply` first")
                raise typer.Exit(1)
            request_id = await cp.invoke_workflow(wf.id, context=context)
            _ok(f"task {request_id}")

    asyncio.run(go())


@app.command()
def tasks(agent: str = typer.Argument(...), limit: int = typer.Option(10, "--limit")) -> None:
    """Recent tasks for an agent."""
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            typer.echo(f"{agent}  workflow={wf.id}  state={wf.lifecycle_state.value}")
            for record in (await cp.get_audit_log(workflow_id=wf.id))[:limit]:
                typer.echo(f"  {getattr(record, 'occurred_at', '')}  "
                           f"{type(record).__name__}  "
                           f"{getattr(record, 'decision', '')}")

    asyncio.run(go())


@app.command()
def status(task_id: str = typer.Argument(..., help="The id `charter run` printed")) -> None:
    """What a task produced, what it cost, and why it stopped."""
    async def go():
        async with _cp() as cp:
            info = await cp.get_request_info(task_id)
            typer.echo(f"state   {getattr(info, 'state', '?')}")
            result = getattr(info, "result", None) or {}
            if result.get("failed"):
                _err(f"failed  {result.get('reason', '')}")
            for key in ("cost_usd", "rounds", "acts_performed", "truncated"):
                if key in result:
                    typer.echo(f"{key:8}{result[key]}")
            for key, value in result.items():
                if key not in ("failed", "reason", "cost_usd", "rounds",
                               "acts_performed", "truncated", "history"):
                    typer.echo(f"{key:8}{value}")

    asyncio.run(go())


@app.command()
def approve(
    approval_id: str = typer.Argument(...),
    agent: str = typer.Option(..., "--agent", "-a"),
    reason: str = typer.Option("", "--reason", "-r", help="Why — recorded in the audit log"),
    actor: str = typer.Option("", "--actor"),
) -> None:
    """Approve a parked gate. `--reason` is worth giving: it lands in the audit log
    and becomes memory the agent reads on its next task."""
    _decide(agent, approval_id, reason, actor, approve=True)


@app.command()
def reject(
    approval_id: str = typer.Argument(...),
    agent: str = typer.Option(..., "--agent", "-a"),
    reason: str = typer.Option("", "--reason", "-r"),
    actor: str = typer.Option("", "--actor"),
) -> None:
    """Reject a parked gate. The reason goes straight back into the agent's next
    round, which is the only reason a rejection teaches it anything."""
    _decide(agent, approval_id, reason, actor, approve=False)


def _decide(agent: str, approval_id: str, reason: str, actor: str, *, approve: bool) -> None:
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            fn = cp.approve_workflow if approve else cp.reject_workflow
            await fn(wf.id, approval_id, actor, reason)
            _ok(("approved " if approve else "rejected ") + approval_id)

    asyncio.run(go())


@app.command()
def answer(
    input_id: str = typer.Argument(...),
    text: str = typer.Argument(...),
    agent: str = typer.Option(..., "--agent", "-a"),
    actor: str = typer.Option("", "--actor"),
) -> None:
    """Answer an agent's question."""
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            await cp.submit_input(wf.id, input_id, {"text": text}, actor)
            _ok(f"answered {input_id}")

    asyncio.run(go())


@app.command()
def worker(
    path: Path = typer.Argument(Path("."), help="worker.yaml or its directory"),
) -> None:
    """Run the generic worker process."""
    try:
        project = _load(path)
    except ConfigError as e:
        _err(f"{len(e.problems)} problem(s):")
        for p in e.problems:
            _err(f"  - {p}")
        raise typer.Exit(1)

    if not hasattr(project, "manifest"):
        _err("worker needs a worker.yaml — point at it or its directory")
        raise typer.Exit(1)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s")

    from .worker import run_worker

    try:
        asyncio.run(run_worker(project))
    except KeyboardInterrupt:
        _ok("stopped")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
