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

from . import ui
from .compile import compile_agent
from .config.loader import ConfigError, load_agent, load_project

app = typer.Typer(
    add_completion=False,
    # No rich panels: plain help reads faster and survives a pipe.
    rich_markup_mode=None,
    no_args_is_help=True,
    # \b is Click's "don't rewrap the next paragraph" marker.
    help="Governed agents from YAML.\n\n\b\n"
         "  charter apply .              create or update agents\n"
         "  charter run <agent> --flag   start one task\n"
         "  charter describe <agent>     state, limits, rules, what's waiting\n"
         "  charter diff .               is what's running what you declared?\n",
)

ENV_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


_err, _warn, _ok = ui.err, ui.warn, ui.ok


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
    try:
        return load_agent(path)
    except ConfigError:
        raise
    except Exception as e:  # noqa: BLE001 — a bug here must still read as a message
        raise ConfigError([f"{path}: could not be read as an agent directory "
                           f"({type(e).__name__}: {e})"]) from e


def _fail(e: ConfigError) -> None:
    ui.err(f"{len(e.problems)} problem(s) in configuration")
    for p in e.problems:
        ui.detail(p)
    raise typer.Exit(1)


@app.command()
def validate(
    path: Path = typer.Argument(Path("."), help="worker.yaml, a project dir, or an agent dir"),
) -> None:
    """Parse and cross-check every file. Touches no network."""
    try:
        loaded = _load(path)
    except ConfigError as e:
        _fail(e)

    bundles = (sorted(loaded.agents.values(), key=lambda b: b.name)
               if hasattr(loaded, "agents") else [loaded])
    ui.table(["name", "versions", "mode", "tools", "gated"], [
        [b.name,
         ",".join(f"v{v}" for v in sorted(b.versions)),
         b.latest.invoke_mode,
         str(len(b.latest.all_tools)),
         ",".join(t.split(".", 1)[1] for t in b.latest.gated_tools) or "-"]
        for b in bundles])


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
        _fail(e)

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
        if cp_cfg.tenant_id:
            project.manifest.control_plane.tenant_id = _resolve(cp_cfg.tenant_id)
    except ConfigError as e:
        for p in e.problems:
            _err(f"  - {p}")
        raise typer.Exit(1)

    async def run() -> None:
        async with ControlPlaneClient(endpoint, api_key) as cp:
            for r in await apply_project(cp, project, only=only):
                ui.ok(f"{ui.ref('agent', r.agent)} v{r.version} "
                      f"{'created' if r.created else 'configured'}")
                for w in r.warnings:
                    ui.warn(f"  warning: {w}")

    _run_apply(run)


def _run_apply(run) -> None:
    from .provisioning.apply import NoSuchTenant
    try:
        asyncio.run(run())
    except NoSuchTenant as e:
        _err(f"no tenant named {e.name!r}")
        _err(f"  existing: {', '.join(e.existing) or '(none)'}")
        _err(f"  create it:  charter tenant create {e.name}")
        raise typer.Exit(1)
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

    from boundflow import ControlPlaneClient

    from .provisioning.apply import apply_bundle, resolve_tenant

    async def run() -> None:
        # Endpoint and key come from BOUNDFLOW_SERVER_ADDRESS / BOUNDFLOW_API_KEY.
        async with ControlPlaneClient() as cp:
            tenant_id = (os.environ.get("BOUNDFLOW_TENANT_ID")
                         or await resolve_tenant(cp, _tenant_name(None)))
            r = await apply_bundle(cp, bundle, tenant_id)
            ui.ok(f"{ui.ref('agent', r.agent)} v{r.version} "
                  f"{'created' if r.created else 'configured'}")
            for w in r.warnings:
                ui.warn(f"  warning: {w}")

    _run_apply(run)


# ── operating an applied agent ──────────────────────────────────────────────
#
# Everything below needs only an agent NAME and BOUNDFLOW_* in the environment —
# no files. The person approving a $240 refund at 2am should not need the repo
# checked out; they got a webhook with an approval_id.


def _cp():
    from boundflow import ControlPlaneClient
    return ControlPlaneClient()


# An agent is identified by (tenant, name) — a workflow's tenant is fixed at
# creation, so the same name in two tenants is two different agents.
TENANT = typer.Option(None, "--tenant", "-t",
                      help="Tenant the agent belongs to [env: CHARTER_TENANT]")


def _tenant_name(explicit: str | None) -> str:
    """No default. Which tenant an agent belongs to is permanent, so it's named
    rather than assumed."""
    name = explicit or os.environ.get("CHARTER_TENANT")
    if not name:
        ui.err("no tenant given")
        ui.detail("--tenant <name>, or export CHARTER_TENANT")
        ui.detail("charter tenant list")
        raise typer.Exit(1)
    return name


async def _workflow_for(cp, agent: str, tenant: str | None = None):
    """Resolve agent name -> workflow within one tenant. Matching on name alone
    would let a staging command act on the production agent."""
    from .provisioning.apply import NoSuchTenant, resolve_tenant

    name = _tenant_name(tenant)
    try:
        tenant_id = await resolve_tenant(cp, name)
    except NoSuchTenant as e:
        ui.err(f"no tenant named {e.name!r}")
        ui.detail(f"existing: {', '.join(e.existing) or '(none)'}")
        raise typer.Exit(1)

    for w in await cp.list_workflows():
        if w.workflow_type == agent and w.tenant_id == tenant_id:
            return w
    return None


def _show_inputs(cfg) -> None:
    """What this agent takes. Printed on the error paths rather than in --help,
    because Typer builds --help before we know which agent was named — and the
    moment someone needs this is the moment they got a flag wrong."""
    if not cfg.inputs:
        typer.echo("  (this agent declares no inputs)")
        return
    typer.echo(f"\ninputs for {cfg.name}:")
    for name, spec in cfg.inputs.items():
        flag = f"--{name.replace('_', '-')}"
        bits = [spec.type]
        if spec.required:
            bits.append("required")
        if spec.default is not None:
            bits.append(f"default {spec.default}")
        if spec.enum:
            bits.append("one of " + "|".join(str(v) for v in spec.enum))
        typer.echo(f"  {flag:<24} {', '.join(bits)}")
        if spec.description:
            typer.echo(f"  {'':<24} {spec.description}")


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
    tenant: str = TENANT,
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
        _show_inputs(cfg)
        raise typer.Exit(1)

    context = {}
    for name, spec in cfg.inputs.items():
        if name in flags:
            context[name] = _coerce(spec, flags[name], name)
        elif spec.default is not None:
            context[name] = spec.default
        elif spec.required:
            _err(f"--{name.replace('_', '-')} is required")
            _show_inputs(cfg)
            raise typer.Exit(1)
        if spec.enum and name in context and context[name] not in spec.enum:
            _err(f"--{name.replace('_', '-')} must be one of {spec.enum}")
            raise typer.Exit(1)

    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant)
            if wf is None:
                _err(f"{agent!r} has not been applied yet — run `charter apply` first")
                raise typer.Exit(1)
            request_id = await cp.invoke_workflow(wf.id, context=context)
            ui.ok(f"{ui.ref('task', request_id)} started")
            ui.detail(f"charter status {request_id}")

    asyncio.run(go())


@app.command()
def describe(agent: str = typer.Argument(...), tenant: str = TENANT) -> None:
    """Everything about one agent, from the control plane alone.

    The screen you run when you get paged: what it is, what it's allowed to spend,
    what will stop it, how close it is to that, and whether anything is waiting on
    you. No checkout — whoever is on call has credentials and a name, not the repo.
    """
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant)
            if wf is None:
                ui.err(f"no agent {agent!r} in tenant {_tenant_name(tenant)}")
                ui.detail("charter agents  # what's there")
                raise typer.Exit(1)
            wf = await cp.get_workflow(wf.id)

            typer.secho(f"{agent}", bold=True)
            ui.kv([("version", f"v{wf.version}"),
                   ("state", ui.state(wf.lifecycle_state.value)),
                   ("workflow", wf.id)], indent="  ")

            # Armed caps. This is the promise — what's enforced should be what the
            # YAML said, and this is where you read it without the YAML.
            policy = await cp.get_agent_runtime_policy(wf.id, agent)
            typer.echo()
            typer.secho("limits per task", fg=typer.colors.BRIGHT_BLACK)
            if policy:
                # Comes back as protobuf-JSON camelCase; show it the way it was
                # written, so what you read here matches runtime.yaml verbatim.
                ui.kv([(_snake(k), _fmt(v)) for k, v in sorted(policy.items())
                       if v not in (0, "", [], None)], indent="  ")
            else:
                ui.detail("none armed")

            stray = await cp.get_agent_lifecycle_policy(wf.id, agent)
            if stray:
                typer.echo()
                ui.err("an agent-lifecycle policy is armed")
                ui.detail("Charter never sets one — a declared cap may be overridden")
                ui.detail(str(stray))

            rules = await cp.get_workflow_lifecycle_policy(wf.id)
            metrics = await cp.get_workflow_metrics(wf.id)
            observed = {
                "num_failures": metrics.total_failures,
                "cost": round(metrics.total_cost_usd, 4),
                "num_llm_calls": metrics.total_llm_calls,
                "latency": round(metrics.total_latency_seconds, 1),
                "approval_rejections": metrics.total_approval_rejections,
            }
            typer.echo()
            typer.secho("rules", fg=typer.colors.BRIGHT_BLACK)
            if not rules:
                ui.detail("none armed")
            labels = [f"{r.metric.value}{f'[{r.tool}]' if r.tool else ''}" for r in rules]
            width = max((len(l) for l in labels), default=0)
            for rule, label in zip(rules, labels):
                action = rule.action.model_dump()
                kind = action.pop("kind", "?")
                detail = " ".join(f"{k}={v}" for k, v in action.items())
                now = (metrics.tool_failure_counts.get(rule.tool, 0) if rule.tool
                       else observed.get(rule.metric.value, 0))
                line = (f"  {label.ljust(width)}   {now} of {rule.threshold:g}"
                        f"   -> {kind} {detail}".rstrip())
                (ui.warn if now >= rule.threshold else typer.echo)(line)

            typer.echo()
            typer.secho("so far", fg=typer.colors.BRIGHT_BLACK)
            ui.kv([("runs", metrics.run_count),
                   ("cost", f"${metrics.total_cost_usd:.4f}"),
                   ("llm calls", metrics.total_llm_calls)], indent="  ")

            if wf.pending_approval:
                g = wf.pending_approval
                ui.gate(agent, "approval", g.approval_id, g.justification, [
                    f"charter approve {g.approval_id} --agent {agent} --reason '...'",
                    f"charter reject  {g.approval_id} --agent {agent} --reason '...'",
                ], timeout=_when(g.timeout_at))
            elif wf.pending_input:
                g = wf.pending_input
                ui.gate(agent, "an answer", g.input_id, g.prompt, [
                    f"charter answer {g.input_id} '...' --agent {agent}"],
                    timeout=_when(g.timeout_at))

    asyncio.run(go())


@app.command()
def tasks(agent: str = typer.Argument(...),
          limit: int = typer.Option(20, "--limit", "-n", help="0 for all"),
          failed: bool = typer.Option(False, "--failed", help="Only runs that didn't succeed"),
          status: str = typer.Option(None, "--status",
                                     help="successful | customer_marked_failure | operation_timeout | ..."),
          since: str = typer.Option(None, "--since", help="24h, 7d, 30m, or 2026-08-01"),
          tenant: str = TENANT,
          path: Path = typer.Option(None, "--path",
                                    help="Where agents live, to show how close rules are to firing")) -> None:
    """An agent's recent tasks, and how close it is to its lifecycle rules.

    The metrics BoundFlow tracks are exactly the ones lifecycle rules evaluate, so
    showing them next to your thresholds answers the question an operator actually
    has — not "what happened" but "is this about to pause itself".
    """
    thresholds = _thresholds(agent, path)

    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant)
            if wf is None:
                _err(f"{agent!r} has not been applied yet — run `charter apply` first")
                raise typer.Exit(1)

            wf = await cp.get_workflow(wf.id)
            state = wf.lifecycle_state.value
            line = f"{agent}  v{wf.version}  {state}"
            (_warn if state in ("blocked", "awaiting_approval", "awaiting_input") else typer.echo)(line)

            m = await cp.get_workflow_metrics(wf.id)
            typer.echo(f"\n  {m.run_count} run(s), ${m.total_cost_usd:.4f}, "
                       f"{m.total_llm_calls} llm calls")
            for metric, value in (("num_failures", m.total_failures),
                                  ("cost", round(m.total_cost_usd, 4)),
                                  ("num_llm_calls", m.total_llm_calls),
                                  ("approval_rejections", m.total_approval_rejections)):
                _rule_line(metric, value, thresholds)
            for tool, count in sorted(m.tool_failure_counts.items()):
                _rule_line("tool_failures", count, thresholds, tool=tool)

            # The API returns every run, newest first, with no server-side filter —
            # so this narrows locally. Correct, but it fetches the whole history to
            # show twenty rows; see the `limit`/cursor ask on the BoundFlow side.
            runs = await cp.list_workflow_runs(wf.id)
            total = len(runs)
            runs = _filter_runs(runs, failed=failed, status=status, since=since)
            shown = runs if limit == 0 else runs[:limit]

            typer.echo()
            if not runs:
                ui.dim(f"no matching tasks ({total} total)")
                return
            ui.table(["task", "outcome", "started", "detail"], [
                [r.request_id,
                 ui.state((r.run_outcome or r.status).value),
                 r.created_at.strftime("%m-%d %H:%M") if r.created_at else "",
                 (r.failure_reason or "")[:60]]
                for r in shown])
            if len(shown) < len(runs):
                ui.dim(f"  {len(shown)} of {len(runs)} matching ({total} total) — -n 0 for all")

    asyncio.run(go())


def _filter_runs(runs: list, *, failed: bool, status: str | None, since: str | None):
    """Narrow a run history. `failed` is the filter people actually reach for —
    "what broke" comes before "what happened on Tuesday"."""
    out = runs
    if failed:
        out = [r for r in out if (r.run_outcome and r.run_outcome.value != "successful")
               or r.status.value == "failed"]
    if status:
        out = [r for r in out
               if (r.run_outcome and r.run_outcome.value == status) or r.status.value == status]
    if since:
        cutoff = _since(since)
        out = [r for r in out if r.created_at and r.created_at.timestamp() >= cutoff]
    return out


def _since(spec: str) -> float:
    """`24h`, `7d`, `30m`, or an ISO date."""
    import datetime as dt

    units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    if spec and spec[-1] in units and spec[:-1].isdigit():
        return dt.datetime.now(dt.timezone.utc).timestamp() - int(spec[:-1]) * units[spec[-1]]
    try:
        parsed = dt.datetime.fromisoformat(spec)
    except ValueError:
        raise typer.BadParameter(f"--since {spec!r}: use 24h, 7d, 30m, or an ISO date")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def _snake(key: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in key)


def _fmt(value):
    """tool_call_limits arrives as a list of dicts; one line each is unreadable."""
    if isinstance(value, list):
        return ", ".join(
            f"{d.get('tool')}={d.get('maxCalls') or d.get('maxFailures')}"
            if isinstance(d, dict) else str(d) for d in value)
    return value


def _when(ts) -> str:
    return ts.strftime("%H:%M") if ts else ""


def _thresholds(agent: str, path: Path | None) -> list:
    """The agent's lifecycle rules, if its files are to hand. Optional — `tasks`
    works without a checkout, it just can't say how close a rule is."""
    if path is None:
        return []
    try:
        loaded = _load(Path(path))
    except ConfigError:
        return []
    bundle = (loaded.agents.get(agent) if hasattr(loaded, "agents") else loaded)
    return list(bundle.lifecycle.rules) if bundle and bundle.lifecycle else []


def _rule_line(metric: str, value, rules: list, tool: str | None = None) -> None:
    label = f"{metric}[{tool}]" if tool else metric
    matching = [r for r in rules if r.when.metric == metric and (r.when.tool == tool)]
    if not matching:
        typer.echo(f"    {label:<28} {value}")
        return
    for rule in matching:
        action = next(k for k in ("pause", "cooldown", "set_version")
                      if getattr(rule.then, k))
        at = rule.when.threshold
        near = value >= at
        text = f"    {label:<28} {value}  (of {at:g} -> {action})"
        (_warn if near else typer.echo)(text)


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
def pending(agent: str = typer.Argument(..., help="Agent name"),
            tenant: str = TENANT) -> None:
    """Show the open gate, if the agent is parked on one.

    This is how you find an approval_id without a webhook — `get_workflow` carries
    the open gate while the workflow is parked, which is exactly the "page reload
    with no in-process state" case.
    """
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            # list_workflows returns a lighter view; the gate only comes back from
            # the single-workflow read.
            wf = await cp.get_workflow(wf.id)

            if wf.pending_approval:
                g = wf.pending_approval
                ui.gate(agent, "approval", g.approval_id, g.justification, [
                    f"charter approve {g.approval_id} --agent {agent} --reason '...'",
                    f"charter reject  {g.approval_id} --agent {agent} --reason '...'",
                ], timeout=_when(g.timeout_at))
            elif wf.pending_input:
                g = wf.pending_input
                ui.gate(agent, "an answer", g.input_id, g.prompt, [
                    f"charter answer {g.input_id} '...' --agent {agent}",
                ], timeout=_when(g.timeout_at))
            else:
                ui.dim(f"{agent}: nothing waiting ({wf.lifecycle_state.value})")

    asyncio.run(go())


@app.command()
def approve(
    approval_id: str = typer.Argument(...),
    agent: str = typer.Option(..., "--agent", "-a"),
    reason: str = typer.Option("", "--reason", "-r", help="Why — recorded in the audit log"),
    actor: str = typer.Option("", "--actor"),
    tenant: str = TENANT,
) -> None:
    """Approve a parked gate. `--reason` is worth giving: it lands in the audit log
    and becomes memory the agent reads on its next task."""
    _decide(agent, approval_id, reason, actor, approve=True, tenant=tenant)


@app.command()
def reject(
    approval_id: str = typer.Argument(...),
    agent: str = typer.Option(..., "--agent", "-a"),
    reason: str = typer.Option("", "--reason", "-r"),
    actor: str = typer.Option("", "--actor"),
    tenant: str = TENANT,
) -> None:
    """Reject a parked gate. The reason goes straight back into the agent's next
    round, which is the only reason a rejection teaches it anything."""
    _decide(agent, approval_id, reason, actor, approve=False, tenant=tenant)


def _decide(agent: str, approval_id: str, reason: str, actor: str, *,
            approve: bool, tenant: str | None = None) -> None:
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            fn = cp.approve_workflow if approve else cp.reject_workflow
            await fn(wf.id, approval_id, actor, reason)
            ui.ok(f"{ui.ref('approval', approval_id)} "
                  f"{'approved' if approve else 'rejected'}")

    asyncio.run(go())


@app.command()
def answer(
    input_id: str = typer.Argument(...),
    text: str = typer.Argument(...),
    agent: str = typer.Option(..., "--agent", "-a"),
    actor: str = typer.Option("", "--actor"),
    tenant: str = TENANT,
) -> None:
    """Answer an agent's question."""
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            await cp.submit_input(wf.id, input_id, {"text": text}, actor)
            ui.ok(f"{ui.ref('input', input_id)} answered")

    asyncio.run(go())


tenant_app = typer.Typer(help="Tenants own agents. One per environment or customer.")
app.add_typer(tenant_app, name="tenant")


@tenant_app.command("list")
def tenant_list() -> None:
    """Tenants in this tenant group."""
    async def go():
        async with _cp() as cp:
            tenants = await cp.list_tenants()
            if not tenants:
                ui.dim("no tenants — charter tenant create default")
                return
            ui.table(["name", "id"], [[t.name, t.id] for t in tenants])

    asyncio.run(go())


@tenant_app.command("create")
def tenant_create(name: str = typer.Argument(..., help="e.g. default, staging, acme-corp")) -> None:
    """Create a tenant. Agents belong to one, and `charter apply` refuses to invent
    one for you — a typo would otherwise mint a second tenant with its own agents
    and its own history, silently."""
    async def go():
        async with _cp() as cp:
            for t in await cp.list_tenants():
                if t.name == name:
                    ui.warn(f"{ui.ref('tenant', name)} already exists")
                    return
            tenant = await cp.create_tenant(name)
            ui.ok(f"{ui.ref('tenant', tenant.name)} created")

    asyncio.run(go())


@app.command()
def diff(
    path: Path = typer.Argument(Path("."), help="worker.yaml or a project dir"),
) -> None:
    """Compare what's armed on the control plane against your files.

    Charter's central promise is that the effective runtime policy always equals
    what runtime.yaml says. This is how you check rather than trust it — including
    that no agent-lifecycle policy exists, since Charter never sets one and
    anything there would silently move a declared cap.
    """
    try:
        project = _load(path)
    except ConfigError as e:
        _fail(e)
    if not hasattr(project, "manifest"):
        _err("diff needs a worker manifest — point at worker.yaml or its directory")
        raise typer.Exit(1)

    async def go():
        drift = 0
        async with _cp() as cp:
            for served in project.manifest.serves:
                bundle = project.agents[served.agent]
                version = max(v for v in served.versions if v in bundle.versions)
                want = compile_agent(bundle, version)

                wf = await _workflow_for(cp, served.agent)
                if wf is None:
                    _warn(f"{served.agent}: not applied")
                    drift += 1
                    continue

                typer.echo(f"\n{served.agent}")
                drift += _diff_line("version", wf.version, want.version)

                live = await cp.get_agent_runtime_policy(wf.id, want.agent_name)
                declared = want.runtime_policy.model_dump(exclude_defaults=True)
                for key, value in declared.items():
                    drift += _diff_line(f"runtime.{key}", _norm(live.get(_camel(key))), _norm(value))

                live_rules = await cp.get_workflow_lifecycle_policy(wf.id)
                drift += _diff_line("lifecycle rules", len(live_rules), len(want.workflow_rules))

                # Charter never sets one. Anything here moved a cap without a version.
                stray = await cp.get_agent_lifecycle_policy(wf.id, want.agent_name)
                if stray:
                    _err(f"  agent-lifecycle policy is set ({stray}) — Charter never "
                         "sets one, so a declared cap may be silently overridden")
                    drift += 1

        if drift:
            _warn(f"\n{drift} difference(s) — `charter apply` to reconcile")
            raise typer.Exit(1)
        _ok("\nin sync")

    asyncio.run(go())


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(w.title() for w in rest)


def _norm(value):
    """Live policy comes back from protobuf JSON, so numbers and lists need
    flattening before they can be compared to what we declared."""
    if isinstance(value, list):
        return sorted(str(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return value


def _diff_line(label: str, live, want) -> int:
    if live == want:
        typer.echo(f"  {label:<26} {want}")
        return 0
    _warn(f"  {label:<26} {want}   (live: {live})")
    return 1


@app.command()
def audit(agent: str = typer.Argument(...), limit: int = typer.Option(20, "--limit"),
          tenant: str = TENANT) -> None:
    """Every governance decision recorded for an agent — who approved what and why,
    and which rule paused it. This is the answer to "prove it did what you say"."""
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            entries = await cp.get_audit_log(workflow_id=wf.id)
            if not entries:
                typer.echo(f"{agent}: nothing recorded yet")
                return
            for e in entries[:limit]:
                when = getattr(e, "occurred_at", None)
                stamp = when.strftime("%Y-%m-%d %H:%M") if when else ""
                if hasattr(e, "approval_id"):
                    who = e.actor or "(no actor)"
                    typer.echo(f"{stamp}  approval {e.decision.value} by {who}")
                    if e.justification:
                        typer.echo(f"                     {e.justification.splitlines()[0]}")
                    if e.reason:
                        typer.echo(f"                     reason: {e.reason}")
                elif hasattr(e, "input_id"):
                    typer.echo(f"{stamp}  input {e.decision.value}: "
                               f"{(e.answer or {}).get('text', '')}")
                else:
                    typer.echo(f"{stamp}  policy fired: {getattr(e, 'metric', '')} -> "
                               f"{getattr(e, 'action', '')}")

    asyncio.run(go())


@app.command()
def resume(agent: str = typer.Argument(...), tenant: str = TENANT) -> None:
    """Release an agent a lifecycle rule paused. Without this a `pause` rule is a
    one-way door."""
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            wf = await cp.get_workflow(wf.id)
            if wf.last_interrupted_request_id:
                await cp.resolve_interrupted_workflow(wf.id, wf.last_interrupted_request_id)
            await cp.activate_workflow(wf.id)
            ui.ok(f"{ui.ref('agent', agent)} active")

    asyncio.run(go())


@app.command()
def delete(
    agent: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation"),
    tenant: str = TENANT,
) -> None:
    """Delete an agent's workflow and its history. Mostly for iterating locally."""
    if not yes:
        typer.confirm(f"Delete {agent} and all its run history?", abort=True)

    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            await cp.delete_workflow(wf.id)
            ui.ok(f"{ui.ref('agent', agent)} deleted")

    asyncio.run(go())


@app.command()
def worker(
    path: Path = typer.Argument(Path("."), help="worker.yaml or its directory"),
) -> None:
    """Run the generic worker process."""
    try:
        project = _load(path)
    except ConfigError as e:
        _fail(e)

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
