"""The `charter` CLI.

A sibling of the `boundflow` CLI, not a layer on top: both are Typer apps calling
ControlPlaneClient over gRPC, and nothing else. It does not open the harness's
store and it does not talk to MCP servers: those are the data plane, the CLI is
an operator tool, and a command that needs a database connection or a vendor
credential is one an operator can't run from where they actually are.

`charter import` used to be the exception — it spawned an MCP server to draft a
config block from its tool list. Authoring convenience, and not worth the hole.

Charter never shells out to `boundflow` — that would
turn validated models into flag strings and typed exceptions into exit codes.

Only `validate` and `apply` live here so far; the rest of the surface (run, status,
approve, pending, memory, worker) lands with the worker.
"""

from __future__ import annotations

import asyncio
import time
import json
import logging
import os
import re
import sys
from pathlib import Path

import typer

from . import ui
from .compile import compile_agent
from .config.agent import split_qualified
try:
    from boundflow.cli.output import is_json, output, set_json
except ImportError:  # pragma: no cover - exercised by the compatibility test
    # The renderer was private as `_output` until boundflow/boundflow#93, which is
    # a rename and a docstring — same three functions either way. Charter names
    # both so it runs against the branch as published today, not only against an
    # unpushed local merge. Delete this once #93 is in a release Charter can pin.
    from boundflow.cli._output import is_json, output, set_json

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
         "  charter agents               what every agent is doing\n"
         "  charter describe <agent>     state, limits, rules, what's waiting\n"
         "  charter diff .               is what's running what you declared?\n",
)

@app.callback()
def _root(
    json_output: bool = typer.Option(
        False, "--json", help="Raw JSON, for scripting"),
) -> None:
    """Charter's own rendering is curated: it shows what an operator on call needs
    and leaves the rest out. `--json` is the escape hatch — BoundFlow's renderer,
    which derives its fields from the record itself, so anything the control plane
    knows is reachable without us having remembered to print it."""
    set_json(json_output)


def _plain(value):
    """Dataclasses to dicts, recursively through lists.

    `output()` converts the object it is handed, not objects nested inside it — so
    a dict of records renders each value through `str()` and you get a repr in your
    JSON. Assembling a composite view means flattening first.
    """
    import dataclasses as dc

    if dc.is_dataclass(value) and not isinstance(value, type):
        return dc.asdict(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _dump(data) -> bool:
    """In --json mode, print the whole record and report that we did.

        if _dump(wf):
            return

    Callers put this above their curated rendering, so the complete record is one
    flag away from every view rather than a thing we re-derive per command.
    """
    if is_json():
        output(data)
        return True
    return False


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


SCHEMAS = {
    "agent": ("charter/agent.schema.json", "agents/*/v*.yaml"),
    "runtime": ("charter/runtime.schema.json", "agents/*/runtime.yaml"),
    "lifecycle": ("charter/lifecycle.schema.json", "agents/*/lifecycle.yaml"),
    "worker": ("charter/worker.schema.json", "worker.yaml"),
}


@app.command()
def schema(
    out: Path = typer.Option(None, "--out", "-o", help="Write all four to this directory"),
    kind: str = typer.Option(None, "--kind", help="agent | runtime | lifecycle | worker"),
) -> None:
    """Emit JSON Schema for the config files.

    Point your editor at it and it does the work the docs otherwise have to: field
    names autocomplete, `metric:` offers the six valid values, a typo is underlined
    before you run anything, and each block carries its own explanation on hover.

        charter schema -o .charter

    then at the top of a file:

        # yaml-language-server: $schema=.charter/agent.schema.json
    """
    from .config.agent import AgentConfig
    from .config.lifecycle import LifecyclePolicyFile
    from .config.runtime import RuntimePolicyFile
    from .config.worker import WorkerManifest

    models = {"agent": AgentConfig, "runtime": RuntimePolicyFile,
              "lifecycle": LifecyclePolicyFile, "worker": WorkerManifest}

    def build(name: str) -> dict:
        doc = models[name].model_json_schema()
        doc["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        doc["$id"] = f"https://charter.dev/schema/v1/{name}.json"
        return doc

    if kind:
        if kind not in models:
            ui.err(f"unknown kind {kind!r} — one of {', '.join(models)}")
            raise typer.Exit(1)
        typer.echo(json.dumps(build(kind), indent=2))
        return

    if out is None:
        ui.err("give --out <dir> to write all four, or --kind <name> for one")
        raise typer.Exit(1)

    out.mkdir(parents=True, exist_ok=True)
    for name in models:
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(build(name), indent=2) + "\n")
        ui.ok(f"{path}")

    typer.echo()
    ui.dim("add to the top of each file (paths are relative to the file):")
    for name, (_, glob) in SCHEMAS.items():
        depth = glob.count("/")
        rel = "../" * depth + f"{out.name}/{name}.schema.json"
        ui.detail(f"{glob:<24} # yaml-language-server: $schema={rel}")


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
         ",".join(split_qualified(t)[1] for t in b.latest.gated_tools) or "-"]
        for b in bundles])


@app.command()
def push(
    agent: str = typer.Argument(..., help="Agent name (its directory)"),
    repository: str = typer.Option(..., "--repository", "-r",
                                   help="e.g. ghcr.io/acme/agents"),
    path: Path = typer.Option(Path("."), "--path", help="Where agents live"),
    version: int = typer.Option(None, "--version", help="Defaults to the newest"),
    insecure: bool = typer.Option(False, "--insecure", help="Plain HTTP (local registries)"),
) -> None:
    """Publish one version's behaviour to a registry.

    What travels is the objective, the tools, the answer shape and the skills —
    everything a worker needs to *be* this agent. Policy is not in here: it
    converges on the control plane through `charter apply`, stays mutable, and
    sealing it would make a thing you can change look like a thing you cannot.

    The result is an ordinary OCI artifact — one gzipped tarball, the same shape a
    Helm chart or an OPA bundle takes. `oras pull` gets the same bytes and `cosign`
    can sign it. Credentials are whatever `docker login` already wrote.

    The tag comes from the config rather than from you. An artifact that said
    version 1 inside and wore a `v2` tag would break the thing the design rests
    on: that a worker can read an artifact and know which handler to register.
    """
    from . import artifact

    try:
        bundle = load_agent(Path(path) / agent)
    except ConfigError as e:
        _fail(e)

    chosen = version or max(bundle.versions)
    if chosen not in bundle.versions:
        _err(f"{agent} has no v{chosen} — it has "
             f"{', '.join(f'v{v}' for v in sorted(bundle.versions))}")
        raise typer.Exit(1)

    packed = artifact.pack(bundle, chosen)

    typer.echo()
    ui.dim(f"packing {agent} v{chosen}")
    for name in packed.files:
        ui.detail(name)
    ui.detail(f"{len(packed.tar):,} bytes  {packed.digest}")

    # An artifact is content-addressed and immutable, which makes it read as a
    # complete description of the agent. For a stdio server it isn't: the config
    # travels, the process does not, and the same digest then behaves differently
    # on two workers — the one thing the artifact exists to prevent. Saying so
    # costs a line; discovering it costs a quarantine at boot, in production.
    local = [m for m in bundle.versions[chosen].mcp if m.command]
    if local:
        typer.echo()
        ui.warn(f"{len(local)} tool server(s) this artifact does not carry:")
        for server in local:
            ui.detail(f"{server.name}: {server.command} {' '.join(server.args)}")
        ui.detail("the worker must provide these — put them in its image, or use "
                  "a url: server and run one as a sidecar")

    typer.echo()
    ui.dim(f"pushing to {packed.reference(repository)}")
    try:
        ref = artifact.push(packed, repository, insecure=insecure)
    except Exception as e:  # noqa: BLE001 — a registry failure is theirs to read
        _err(f"push failed: {type(e).__name__}: {e}")
        ui.detail("credentials come from `docker login <registry>`")
        raise typer.Exit(1) from e

    typer.echo()
    _ok(ref)
    ui.detail(f"serve it:  charter worker  (serves: [{ref.split('@')[0]}])")


@app.command()
def apply(
    path: Path = typer.Argument(Path("."), help="worker.yaml or a project dir"),
    only: str = typer.Option(None, "--agent", help="Apply just this agent"),
    instance: str = typer.Option(None, "--instance", help="Configure one instance"),
    all_: bool = typer.Option(False, "--all", help="Configure every instance"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print, don't call"),
) -> None:
    """Validate, then update config, policies, and pricing on existing instances.

    Never creates or destroys one: an instance is an entity with state of its own,
    and bringing one into existence shouldn't be a side effect of re-running config
    in CI. See `charter agent create`.

    With more than one instance you must say which — `--instance <id>` or `--all` —
    because each has its own state and quietly configuring the wrong one is the
    failure this refuses to allow.
    """
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
            _print_compiled(compile_agent(bundle, version))
        ui.dim("\ndry run — nothing applied")
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
            results = await apply_project(cp, project, only=only,
                                          instance=instance, all_=all_)
            if not results:
                ui.warn("nothing to configure — no instances exist yet")
                ui.detail("charter agent create <agent>")
                return
            for r in results:
                where = f" {short(r.workflow_id)}" if r.workflow_id else ""
                ui.ok(f"{ui.ref('agent', r.agent)}{where} v{r.version} configured")
                for w in r.warnings:
                    ui.warn(f"  warning: {w}")

    _run_apply(run)


def _run_apply(run) -> None:
    from .provisioning.apply import AmbiguousInstance, NoSuchTenant
    try:
        asyncio.run(run())
    except AmbiguousInstance as e:
        ui.err(f"{e.agent!r} has {len(e.ids)} instances — say which")
        for wid in e.ids:
            ui.detail(short(wid))
        typer.echo()
        ui.detail("--instance <id>   configure one")
        ui.detail("--all             configure every instance")
        raise typer.Exit(1)
    except NoSuchTenant as e:
        _err(f"no tenant named {e.name!r}")
        _err(f"  existing: {', '.join(e.existing) or '(none)'}")
        _err(f"  create it:  charter tenant create {e.name}")
        raise typer.Exit(1)
    except Exception as e:  # noqa: BLE001 — the CLI is the boundary; show it plainly
        _err(f"apply failed: {type(e).__name__}: {e}")
        raise typer.Exit(1)


def _print_compiled(c) -> None:
    w = c.workflow_config
    typer.secho(f"\n{c.name} v{c.version}", bold=True)
    ui.kv(_config_lines(w), indent="  ")
    caps = c.runtime_policy.model_dump(exclude_defaults=True)
    ui.kv([(_snake(k), _fmt(v)) for k, v in sorted(caps.items())], indent="  ")
    for rule in c.workflow_rules:
        action = rule.action.model_dump()
        kind = action.pop("kind", "?")
        detail = " ".join(f"{k}={v}" for k, v in action.items())
        typer.echo(f"  rule                 {rule.metric.value} >= {rule.threshold:g}"
                   f" -> {kind} {detail}".rstrip())


def _apply_single(bundle, *, dry_run: bool) -> None:
    """Apply one agent directory using BOUNDFLOW_* from the environment."""
    compiled = compile_agent(bundle)
    if dry_run:
        _print_compiled(compiled)
        ui.dim("\ndry run — nothing applied")
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


def _manifest():
    """The worker manifest for the project we're standing in, or None.

    `CHARTER_PROJECT`, else `worker.yaml` here. Never an error: most commands work
    without a checkout at all — whoever is on call has credentials and an agent
    name, not the repo — so a missing or unreadable manifest just means we fall
    back to the environment.
    """
    where = os.environ.get("CHARTER_PROJECT") or "worker.yaml"
    path = Path(where)
    if path.is_dir():
        path = path / "worker.yaml"
    if not path.is_file():
        return None
    try:
        project = load_project(path)
    except Exception as e:  # noqa: BLE001 — a broken file is not a dead command
        # Falling back silently is how you end up pointed at the wrong control
        # plane: the file names an endpoint, it doesn't parse, and the command
        # quietly uses whatever the environment had. Say so and carry on.
        ui.warn(f"{path} could not be read — using the environment instead")
        ui.detail(str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__)
        ui.detail("charter validate  # for the whole list")
        return None
    return getattr(project, "manifest", None)


def _cp():
    """A control-plane client, configured the same way the worker configures its own.

    The manifest wins where it says something. Without it, `ControlPlaneClient()`
    reads the environment, which is what an on-call operator with no checkout has.

    This used to be bare `ControlPlaneClient()`, so a key written into worker.yaml
    worked for `charter worker` and not for any other command — the config file
    said where the control plane was and every read command ignored it.
    """
    from boundflow import ControlPlaneClient
    from .worker import resolve

    manifest = _manifest()
    if manifest is None:
        return ControlPlaneClient()
    try:
        return ControlPlaneClient(resolve(manifest.control_plane.endpoint),
                                  resolve(manifest.control_plane.api_key))
    except RuntimeError:
        # A ${VAR} the manifest names and the environment doesn't have. The
        # environment may still carry BOUNDFLOW_* directly, so this is not fatal
        # here — it is `charter worker` that must refuse.
        return ControlPlaneClient()


# An agent is identified by (tenant, name) — a workflow's tenant is fixed at
# creation, so the same name in two tenants is two different agents.
INSTANCE = typer.Option(
    None, "--instance",
    help="Which instance, when the agent has several")

TENANT = typer.Option(None, "--tenant", "-t",
                      help="Tenant the agent belongs to [env: CHARTER_TENANT]")


def _tenant_name(explicit: str | None) -> str:
    """No default. Which tenant an agent belongs to is permanent, so it's named
    rather than assumed."""
    manifest = _manifest()
    name = (explicit or os.environ.get("CHARTER_TENANT")
            or (manifest.control_plane.tenant if manifest else ""))
    if not name:
        ui.err("no tenant given")
        ui.detail("--tenant <name>, export CHARTER_TENANT, or name one in "
                  "worker.yaml")
        ui.detail("charter tenant list")
        raise typer.Exit(1)
    return name


def short(workflow_id: str) -> str:
    """The first 8 characters of a workflow id.

    An instance is identified by the id BoundFlow assigned it — there is nowhere
    else to put a name, and Charter deliberately keeps no store to map one. An
    ordinal would be worse than ugly: delete instance 2 and either 3 becomes 2,
    silently repointing anything that referred to it, or you keep gaps and the
    numbers stop being ordinals. A short id is derived from the real one, so it
    never moves.
    """
    return workflow_id[:8]


async def _tenant_id(cp, tenant: str | None):
    from .provisioning.apply import NoSuchTenant, resolve_tenant

    name = _tenant_name(tenant)
    try:
        return await resolve_tenant(cp, name)
    except NoSuchTenant as e:
        ui.err(f"no tenant named {e.name!r}")
        ui.detail(f"existing: {', '.join(e.existing) or '(none)'}")
        raise typer.Exit(1)


async def _instances(cp, agent: str, tenant: str | None = None) -> list:
    """Every live instance of an agent, in creation order.

    Scoped by tenant, because identity is (tenant, name) and a workflow's tenant is
    fixed at creation. Matching on name alone would let a staging command act on
    production.
    """
    tid = await _tenant_id(cp, tenant)
    return [w for w in await cp.list_workflows()
            if w.workflow_type == agent and w.tenant_id == tid]


async def _select(cp, agent: str, *, instance: str | None, all_: bool,
                  tenant: str | None = None, verb: str = "act on",
                  fans_out: bool = True) -> list:
    """Which instances a command should touch, refusing to guess.

    An agent can have several instances, and each is a distinct entity with its own
    state. Picking one for someone would silently send work to the wrong entity's
    memory, so with more than one and no choice made, this stops.
    """
    found = await _instances(cp, agent, tenant)
    if not found:
        ui.err(f"no instances of {agent!r}")
        ui.detail(f"charter agent create {agent}")
        raise typer.Exit(1)

    if instance:
        matched = [w for w in found if w.id.startswith(instance)]
        if not matched:
            ui.err(f"no instance of {agent!r} starting {instance!r}")
            ui.detail("known: " + ", ".join(short(w.id) for w in found))
            raise typer.Exit(1)
        if len(matched) > 1:
            ui.err(f"{instance!r} matches {len(matched)} instances — use more characters")
            ui.detail("  " + ", ".join(short(w.id) for w in matched))
            raise typer.Exit(1)
        return matched

    if all_:
        return found

    # Not even when there is only one. An instance is the entity that holds the
    # state — the conversation, the budget, the lifecycle history — and an agent
    # name addresses a *kind* of it. Resolving the name whenever the count happens
    # to be one teaches that the name is the thing, and the lesson expires silently
    # the day someone creates a second: the same command that worked all year now
    # means something else, or stops working, depending on which command it was.
    # Being explicit costs a flag; being implicit costs it exactly when an agent
    # has become important enough to run twice.
    ui.err(f"{agent!r} has {len(found)} "
           f"{'instance' if len(found) == 1 else 'instances'} — say which")
    for w in found:
        ui.detail(f"{short(w.id)}  v{w.version}  {_state_of(w)}")
    typer.echo()
    ui.detail(f"--instance {short(found[0].id)}   {verb} that one")
    if fans_out:
        ui.detail(f"--all             {verb} every instance")
    raise typer.Exit(1)


def _state_of(w) -> str:
    state = getattr(w, "workflow_state", None)
    return getattr(state, "value", state) or "unknown"


async def _workflow_for(cp, agent: str, tenant: str | None = None,
                        instance: str | None = None, verb: str = "act on"):
    """The one instance a command should act on.

    Every read and every operator action needs this, and every one of them was
    silently taking whichever instance came back first — which is fine until an
    agent has two, at which point `charter resume` releases the wrong entity.
    """
    found = await _instances(cp, agent, tenant)
    if not found:
        return None
    return (await _select(cp, agent, instance=instance, all_=False,
                          tenant=tenant, verb=verb, fans_out=False))[0]


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
    instance: str = typer.Option(None, "--instance", help="Which instance to run on"),
    all_: bool = typer.Option(False, "--all", help="Start a task on every instance"),
    tenant: str = TENANT,
) -> None:
    """Start one task. Declared inputs become --flags, validated before the request
    is created so a typo fails here instead of burning a run.

    An agent with several instances needs one naming: each has its own state, so
    sending work to the wrong one isn't a scheduling detail, it's the wrong entity
    doing the job.
    """
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
            targets = await _select(cp, agent, instance=instance, all_=all_,
                                    tenant=tenant, verb="run on")
            for wf in targets:
                request_id = await cp.invoke_workflow(wf.id, context=context)
                where = f" on {short(wf.id)}" if len(targets) > 1 else ""
                ui.ok(f"{ui.ref('task', request_id)} started{where}")
                if len(targets) == 1:
                    ui.detail(f"charter status {request_id}")

    asyncio.run(go())


agent_app = typer.Typer(help="Create and destroy instances of an agent.")
app.add_typer(agent_app, name="agent")


@agent_app.command("create")
def agent_create(
    agent: str = typer.Argument(..., help="Agent name (its directory)"),
    path: Path = typer.Option(Path("."), "--path", help="Where agents live"),
    tenant: str = TENANT,
) -> None:
    """Bring a new instance into existence.

    Separate from `apply` on purpose. An instance is an entity with its own store,
    its own budget and its own lifecycle state, so creating one is a decision
    someone makes rather than something a config run does on their behalf.
    """
    from .compile import compile_agent
    from .provisioning.apply import create_instance

    try:
        bundle = load_agent(Path(path) / agent)
    except ConfigError as e:
        _fail(e)

    async def go():
        async with _cp() as cp:
            tid = await _tenant_id(cp, tenant)
            existing = await _instances(cp, agent, tenant)
            wf = await create_instance(cp, compile_agent(bundle), tid)
            ui.ok(f"{ui.ref('agent', agent)} {short(wf.id)} created")
            if existing:
                ui.detail(f"{len(existing) + 1} instances — "
                          f"`charter run {agent} --instance {short(wf.id)}`")

    asyncio.run(go())


@agent_app.command("delete")
def agent_delete(
    agent: str = typer.Argument(..., help="Agent name"),
    instance: str = typer.Option(..., "--instance", help="Which instance to destroy"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation"),
    wait: bool = typer.Option(False, "--wait", help="Block until the delete completes"),
    tenant: str = TENANT,
) -> None:
    """Destroy one instance, and everything it remembered.

    Requires `--instance` even when there is only one: this is the operation that
    loses state irrecoverably, so it should never be possible to run it against
    whichever instance happened to come back first.
    """
    async def go():
        async with _cp() as cp:
            targets = await _select(cp, agent, instance=instance, all_=False,
                                    tenant=tenant, verb="delete")
            wf = targets[0]
            if not yes:
                ui.warn(f"deleting {agent} {short(wf.id)} — its memory and files go "
                        f"with it, and nothing restores them")
                typer.confirm("continue?", abort=True)
            await cp.delete_workflow(wf.id)

            # The control plane won't finish a delete while a run is in flight — it
            # sits blocked until the task completes. That's the right behaviour, and
            # worth saying out loud rather than leaving someone to find a workflow
            # that didn't disappear.
            after = await cp.get_workflow(wf.id)
            if after.lifecycle_state.value == "deleted":
                ui.ok(f"{ui.ref('agent', agent)} {short(wf.id)} deleted")
                return

            ui.ok(f"{ui.ref('agent', agent)} {short(wf.id)} delete queued")
            ui.detail("a task is still running — it finishes first, then the "
                      "instance goes")
            if not wait:
                return

            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                await asyncio.sleep(2)
                if (await cp.get_workflow(wf.id)).lifecycle_state.value == "deleted":
                    ui.ok(f"{short(wf.id)} deleted")
                    return
            # An abandoned job's lease expires and any worker serving this type
            # picks it up — so this only persists when nothing serves it any more.
            ui.warn("still running after 5m — is a worker still serving this agent?")

    asyncio.run(go())


@app.command()
def agents(tenant: str = TENANT) -> None:
    """Every agent in a tenant and what it's doing.

    Both states, because they answer different questions: `status` is whether a
    rule has stopped it, `activity` is whether it's waiting on you. An agent can be
    active and blocked on a human, or paused and idle, and only one of those is
    something you fix by approving something.
    """
    async def go():
        from .provisioning.apply import NoSuchTenant, resolve_tenant

        async with _cp() as cp:
            name = _tenant_name(tenant)
            try:
                tenant_id = await resolve_tenant(cp, name)
            except NoSuchTenant as e:
                ui.err(f"no tenant named {e.name!r}")
                ui.detail(f"existing: {', '.join(e.existing) or '(none)'}")
                raise typer.Exit(1)

            mine = [w for w in await cp.list_workflows()
                    if w.tenant_id == tenant_id and w.lifecycle_state.value != "deleted"]
            if _dump(mine):
                return
            if not mine:
                ui.dim(f"no agents in {name} — charter apply .")
                return

            mine.sort(key=lambda w: (w.workflow_type, w.id))
            # An agent can have several instances — same config, separate entities,
            # each with its own state and its own budget. The name is printed once
            # and the instance id identifies the row, because two rows reading
            # `refund-demo v1 active` tell you nothing about which is stuck.
            rows, seen = [], None
            for w in mine:
                rows.append([w.workflow_type if w.workflow_type != seen else "",
                             short(w.id), f"v{w.version}",
                             ui.state(w.workflow_state.value),
                             ui.state(ui.activity(w.lifecycle_state.value))])
                seen = w.workflow_type
            # No schedule column: list_workflows returns the lighter view with
            # config unset, and a fleet view that made an extra call per agent to
            # fill one column would be the wrong trade. `charter describe` has it.
            ui.table(["agent", "instance", "ver", "status", "activity"], rows)

            # The two reasons an agent isn't working, and they need different acts.
            waiting = [w for w in mine
                       if w.lifecycle_state.value in ("awaiting_approval", "awaiting_input")]
            stopped = [w for w in mine if not ui.working(w.workflow_state.value)]

            if waiting:
                typer.echo()
                ui.warn(f"{len(waiting)} waiting on a human")
                for w in waiting:
                    ui.detail(f"charter pending {w.workflow_type} --instance {short(w.id)}")
            if stopped:
                typer.echo()
                ui.warn(f"{len(stopped)} stopped — no new tasks will start")
                for w in stopped:
                    ui.detail(f"charter audit {w.workflow_type} --instance {short(w.id)}")
                    ui.detail(f"charter resume {w.workflow_type} --instance {short(w.id)}")

    asyncio.run(go())


@app.command()
def describe(
    agent: str = typer.Argument(...),
    tenant: str = TENANT,
    instance: str = INSTANCE,
) -> None:
    """Everything about one agent, from the control plane alone.

    The screen you run when you get paged: what it is, what it's allowed to spend,
    what will stop it, how close it is to that, and whether anything is waiting on
    you. No checkout — whoever is on call has credentials and a name, not the repo.
    """
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant, instance)
            if wf is None:
                ui.err(f"no agent {agent!r} in tenant {_tenant_name(tenant)}")
                ui.detail("charter agents  # what's there")
                raise typer.Exit(1)
            wf = await cp.get_workflow(wf.id)

            if is_json():
                # Everything the control plane holds about this agent, assembled
                # rather than summarised. The curated view below is the one to read
                # at 3am; this is the one to pipe into something.
                output({
                    "workflow": _plain(wf),
                    "runtime_policy": await cp.get_agent_runtime_policy(wf.id, agent),
                    "lifecycle_rules": _plain(
                        await cp.get_workflow_lifecycle_policy(wf.id)),
                    "metrics": _plain(await cp.get_workflow_metrics(wf.id)),
                })
                return

            typer.secho(f"{agent}", bold=True)
            ui.kv([("version", f"v{wf.version}"),
                   # workflow_state is the one that decides whether tasks start;
                   # lifecycle_state is only where it happens to be right now.
                   ("status", ui.state(wf.workflow_state.value)),
                   ("activity", ui.state(ui.activity(wf.lifecycle_state.value))),
                   ("workflow", wf.id)], indent="  ")

            # The control plane records more about a held or dying workflow than
            # "paused" conveys, and this is the screen you run when you get paged.
            # Leaving it out means the two questions you actually have — whose hold
            # is this, and can I release it yet — are answerable only by reading the
            # API directly, which is the opposite of what this command promises.
            if wf.deletion_requested_at:
                typer.echo()
                ui.err("deletion requested")
                ui.detail(f"at {_stamp(wf.deletion_requested_at)}")
                ui.detail("it is going away; work started now may not survive")

            if wf.suspension is not None:
                held = wf.suspension
                typer.echo()
                typer.secho("hold", fg=typer.colors.BRIGHT_BLACK)
                ui.kv([("id", held.suspension_id),
                       ("reason", held.reason or "none given"),
                       ("placed", _stamp(held.requested_at)),
                       ("stopping the run", "yes" if held.stop_current else "no"),
                       # Why this line is here rather than only discovered at
                       # resume: until it drains, `charter resume` refuses, and
                       # "resume is broken" is the wrong conclusion to reach alone
                       # at 3am.
                       ("drained", "yes" if held.finalized_at else
                        "no — cannot be resumed yet")], indent="  ")
                ui.detail(f"charter resume {agent} --suspension {held.suspension_id}")

            if wf.last_interrupted_request_id:
                typer.echo()
                ui.err("a run was interrupted by the platform")
                ui.detail(f"task {short(wf.last_interrupted_request_id)}")
                ui.detail(f"charter resume {agent}  # resolves it")

            typer.echo()
            typer.secho("configuration", fg=typer.colors.BRIGHT_BLACK)
            ui.kv(_config_lines(wf.config), indent="  ")
            if not ui.working(wf.workflow_state.value):
                ui.detail("no new tasks will start — charter resume "
                          f"{agent}" if wf.workflow_state.value == "paused" else "")

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
          instance: str = INSTANCE,
          # `took` is wall clock and includes time parked at a gate.
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
            wf = await _workflow_for(cp, agent, tenant, instance)
            if wf is None:
                _err(f"{agent!r} has not been applied yet — run `charter apply` first")
                raise typer.Exit(1)

            wf = await cp.get_workflow(wf.id)
            line = (f"{agent}  v{wf.version}  {ui.state(wf.workflow_state.value)}"
                    f"  {ui.state(ui.activity(wf.lifecycle_state.value))}")
            typer.echo(line)
            if not ui.working(wf.workflow_state.value):
                ui.warn(f"  stopped — no new tasks will start")

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
            if _dump(runs):
                return
            total = len(runs)
            runs = _filter_runs(runs, failed=failed, status=status, since=since)
            shown = runs if limit == 0 else runs[:limit]

            typer.echo()
            if not runs:
                ui.dim(f"no matching tasks ({total} total)")
                return
            reasons = await _reasons(cp, shown)
            ui.table(["task", "outcome", "started", "took"],
                     [[r.request_id,
                       ui.state((r.run_outcome or r.status).value),
                       r.created_at.strftime("%m-%d %H:%M") if r.created_at else "",
                       _took(r.created_at, r.completed_at)] for r in shown],
                     notes=[_first_line(reasons.get(r.request_id, r.failure_reason))
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


def _config_lines(cfg) -> list[tuple[str, object]]:
    """The workflow config as a customer reads it. Shared by `describe`, which reads
    it from the control plane, and `apply --dry-run`, which compiles it from files —
    so what you're about to apply and what is running are described the same way.

    None when the control plane has no config for this workflow. That is real —
    `list_workflows` returns a lighter record without one — and it reached here as
    an AttributeError on `repeat_every_seconds`, which reads as a Charter crash
    rather than the absence it is."""
    if cfg is None:
        return [("config", "none on the control plane — charter apply")]
    schedule = (f"every {_duration(cfg.repeat_every_seconds)}"
                if cfg.repeat_every_seconds else "on demand")
    if not cfg.triggerable:
        schedule += ", no manual runs"
    queued = (f", max {cfg.max_queue_depth} queued" if cfg.max_queue_depth
              else ", server default depth" if cfg.invoke_mode.value == "queue" else "")
    return [("runs", schedule),
            ("piled-up invokes", cfg.invoke_mode.value + queued),
            ("if a worker dies", "another picks it up" if cfg.resumable
             else "the workflow is interrupted until someone clears it"),
            # Not "round deadline": `round` is an internal unit, and the whole
            # point of replacing max_iterations with drafts/questions/tool-failures
            # was that nobody should have to know what one is.
            ("cancelled after", _duration(cfg.invoke_timeout_seconds))]


def _stamp(ts) -> str:
    """A timestamp to the second, date included — `_when` gives clock time only,
    which is ambiguous for a hold placed yesterday. "-" for None, because "never"
    and "midnight 1970" are not the same thing."""
    return ts.isoformat(sep=" ", timespec="seconds") if ts else "-"


def _snake(key: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in key)


def _fmt(value):
    """tool limits arrive as a list of dicts — camelCase from protobuf JSON on the
    read path, snake_case from a pydantic dump on the compile path."""
    if isinstance(value, list):
        return ", ".join(_one_limit(d) if isinstance(d, dict) else str(d) for d in value)
    return value


def _one_limit(d: dict) -> str:
    n = next((d[k] for k in ("maxCalls", "max_calls", "maxFailures", "max_failures")
              if d.get(k) is not None), "?")
    return f"{d.get('tool')}={n}"


def _took(started, finished) -> str:
    """Wall clock, so it includes time parked at a gate — a task that waited 30
    minutes for an approval reads 31m even though it worked for one."""
    if not (started and finished):
        return ""
    return _duration(int((finished - started).total_seconds()))


async def _reasons(cp, runs: list) -> dict[str, str]:
    """Why each failed run failed, for the ones that don't say so themselves.

    `failure_reason` is the control plane's field, and it carries platform
    failures — a timeout, a worker that died. A task Charter itself marked failed
    publishes its reason in the result instead, so that column comes back empty
    for exactly the failures Charter is responsible for explaining. Reading the
    result is the only way to get it.

    One extra call per failed run, and only for the rows about to be printed: a
    healthy history costs nothing, and a broken one is worth a few round trips to
    avoid sending someone off to run `charter status` by hand.
    """
    out: dict[str, str] = {}
    for run in runs:
        outcome = (run.run_outcome.value if run.run_outcome else "")
        if run.failure_reason or "fail" not in outcome:
            continue
        try:
            info = await cp.get_request_info(run.request_id)
        except Exception:  # noqa: BLE001 — a listing must not die explaining itself
            continue
        reason = (info.result or {}).get("reason")
        if reason:
            out[run.request_id] = reason
    return out


def _first_line(reason: str) -> str:
    """The first line, whole. ui.table fits it to the terminal; `charter status`
    prints the entire reason including the rest of a traceback."""
    return reason.strip().splitlines()[0] if reason else ""


def _duration(seconds: int) -> str:
    """Back to how it was written — nobody thinks in seconds past a minute."""
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            whole, rest = divmod(seconds, size)
            return f"{whole}{unit}" if not rest or seconds >= 3600 else f"{whole}{unit}{rest}s"
    return f"{seconds}s"


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
            if _dump(info):
                return
            outcome = info.run_outcome.value if info.run_outcome else info.status.value
            ui.kv([("task", task_id),
                   ("outcome", ui.state(outcome)),
                   ("started", info.created_at.strftime("%Y-%m-%d %H:%M:%S") if info.created_at else ""),
                   ("took", _took(info.created_at, info.completed_at) or "-")])

            # An uncaught exception never got far enough to publish a result, so
            # failure_reason is the only record of it. Printed whole — a truncated
            # stack trace is the one thing nobody wants.
            if info.failure_reason:
                typer.echo()
                ui.err("failed")
                for line in info.failure_reason.splitlines():
                    ui.detail(line)

            if info.invoke_context:
                given = {k: v for k, v in info.invoke_context.items() if not k.startswith("_")}
                if given:
                    typer.echo()
                    typer.secho("inputs", fg=typer.colors.BRIGHT_BLACK)
                    ui.kv(sorted(given.items()), indent="  ")

            result = info.result or {}
            if not result:
                return

            # What a failed task reports. A successful one publishes the agent's own
            # answer and nothing else, so on that path there is no spend to strip.
            spent = ("cost_usd", "llm_calls", "gates", "seconds")
            typer.echo()
            if result.get("failed"):
                ui.err(result.get("reason", "failed"))
            else:
                typer.secho("result", fg=typer.colors.BRIGHT_BLACK)
                ui.kv([(k, v) for k, v in result.items()
                       if k not in ("failed", "reason", *spent)], indent="  ")

            if any(k in result for k in spent):
                typer.echo()
                typer.secho("spent", fg=typer.colors.BRIGHT_BLACK)
                ui.kv([(k, result[k]) for k in spent if k in result], indent="  ")

    asyncio.run(go())


@app.command()
def pending(agent: str = typer.Argument(..., help="Agent name"),
    instance: str = INSTANCE,
            tenant: str = TENANT) -> None:
    """Show the open gate, if the agent is parked on one.

    This is how you find an approval_id without a webhook — `get_workflow` carries
    the open gate while the workflow is parked, which is exactly the "page reload
    with no in-process state" case.
    """
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant, instance)
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
    instance: str = INSTANCE,
    tenant: str = TENANT,
) -> None:
    """Approve a parked gate. `--reason` is worth giving: it lands in the audit log
    and becomes memory the agent reads on its next task."""
    _decide(agent, approval_id, reason, actor, approve=True, tenant=tenant,
            instance=instance)


@app.command()
def reject(
    approval_id: str = typer.Argument(...),
    agent: str = typer.Option(..., "--agent", "-a"),
    reason: str = typer.Option("", "--reason", "-r"),
    actor: str = typer.Option("", "--actor"),
    instance: str = INSTANCE,
    tenant: str = TENANT,
) -> None:
    """Reject a parked gate. The reason goes straight back into the agent's next
    round, which is the only reason a rejection teaches it anything."""
    _decide(agent, approval_id, reason, actor, approve=False, tenant=tenant,
            instance=instance)


def _decide(agent: str, approval_id: str, reason: str, actor: str, *,
            approve: bool, tenant: str | None = None,
            instance: str | None = None) -> None:
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant, instance)
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
    instance: str = INSTANCE,
) -> None:
    """Answer an agent's question."""
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant, instance)
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
def audit(
    agent: str = typer.Argument(...),
    limit: int = typer.Option(20, "--limit"),
    instance: str = INSTANCE,
    tenant: str = TENANT,
) -> None:
    """Every governance decision recorded for an agent — who approved what and why,
    and which rule paused it. This is the answer to "prove it did what you say"."""
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant, instance)
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
def pause(
    agent: str = typer.Argument(...),
    tenant: str = TENANT,
    instance: str = INSTANCE,
    now: bool = typer.Option(False, "--now", help="Stop the run in flight too"),
    reason: str = typer.Option("", "--reason", help="Recorded on the hold"),
) -> None:
    """Stop an agent taking new work, and hold what is queued.

    The hold is an operator's, not a rule's: nothing about the agent's config or
    its lifecycle policy changes, and `charter resume` puts it back exactly as it
    was. Queued tasks are kept, not dropped — see `charter abandon` for that, which
    is a separate command because it cannot be undone.

    By default a task already running is left to finish, which is usually what you
    want: it may be mid-conversation, and killing it loses that turn. `--now` asks
    for it to stop instead, which is best-effort — a run that finishes first
    finishes.
    """
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant, instance, verb="pause")
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)

            full = await cp.get_workflow(wf.id)
            held = full.suspension
            if held is not None:
                # `--now` on an existing hold is an escalation, not a repeat: a
                # graceful pause left the run alone, and this is asking for it to
                # stop. Reporting "already paused" and returning would drop that
                # request on the floor — the operator's second, more urgent command
                # doing strictly less than their first.
                if not now or held.stop_current:
                    ui.dim(f"{agent} is already paused")
                    ui.detail(f"reason: {held.reason or 'none given'}")
                    if now and held.stop_current:
                        ui.detail("the run in flight is already being stopped")
                    return
                if held.finalized_at is not None:
                    ui.dim(f"{agent} is already paused, and nothing is running")
                    ui.detail("--now has nothing to stop")
                    return
                # Retarget the hold rather than opening a second one, so `resume`
                # still has one suspension_id to release. Last-write-wins, so the
                # existing reason has to be carried or it would be overwritten with
                # the empty string.
                await cp.suspend_workflow(wf.id, reason=reason or held.reason,
                                          stop_current_run=True,
                                          suspension_id=held.suspension_id)
                _ok(f"{ui.ref('agent', agent)} hold escalated")
                ui.detail(f"hold {held.suspension_id}")
                ui.detail("the run in flight is being stopped")
                ui.detail(f"charter resume {agent} --suspension "
                          f"{held.suspension_id}")
                return

            sid = await cp.suspend_workflow(wf.id, reason=reason,
                                            stop_current_run=now)
            _ok(f"{ui.ref('agent', agent)} paused")
            # The id identifies *your* hold, and is the whole reason the call
            # returns one. Printing it is what lets you later release the hold you
            # placed rather than whichever one the server happens to report — which
            # may be someone else's, placed for a reason you can't see.
            ui.detail(f"hold {sid}")
            if full.lifecycle_state.value in ("awaiting_approval", "awaiting_input"):
                # Worth saying now rather than letting it be discovered at resume:
                # a task parked on a human holds the suspension open indefinitely,
                # and `--now` does not help, because there is no run to stop.
                _warn("a task is waiting on you, and the hold can't be released "
                      "until it's answered")
                ui.detail(f"charter pending {agent} --instance {short(wf.id)}")
                return
            # The hold is recorded immediately; a run in flight drains behind it,
            # and until that finishes the workflow cannot be resumed. Saying so
            # here is the difference between "wait a moment" and "resume is broken".
            ui.detail("the run in flight is being stopped" if now
                      else "a run already in flight will finish first")
            ui.detail(f"charter resume {agent} --suspension {sid}")

    asyncio.run(go())


@app.command()
def abandon(
    agent: str = typer.Argument(...),
    tenant: str = TENANT,
    instance: str = INSTANCE,
    task: str = typer.Option(None, "--task", help="One queued task id"),
    all_: bool = typer.Option(False, "--all", help="Every queued task"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation"),
) -> None:
    """Drop queued tasks that have not started. Cannot be undone.

    Only ever touches work that is still waiting: a task already running or
    scheduled is left alone, so this is safe to call at any moment. Which is also
    why the count it reports can be lower than what you asked for — anything that
    started in between stayed.

    Separate from `charter pause` on purpose. Pausing is reversible and abandoning
    is not, and one flag away from each other is too close for a thing you cannot
    take back.
    """
    if not task and not all_:
        _err("say which — --task <id>, or --all")
        raise typer.Exit(1)

    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant, instance, verb="abandon work for")
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)

            if not yes:
                what = "every queued task" if all_ else f"task {short(task)}"
                typer.confirm(f"abandon {what} for {agent}? this cannot be undone",
                              abort=True)

            dropped = await cp.abandon_queued_requests(
                wf.id, request_ids=None if all_ else [task], all=all_)
            if not dropped:
                ui.dim("nothing was queued — anything running or scheduled is "
                       "untouched by this")
                return
            _ok(f"abandoned {len(dropped)} queued task(s)")
            for request_id in dropped:
                ui.detail(short(request_id))

    asyncio.run(go())


@app.command()
def resume(
    agent: str = typer.Argument(...),
    tenant: str = TENANT,
    instance: str = INSTANCE,
    suspension: str = typer.Option("", "--suspension", help=(
        "The hold to release — the id `charter pause` printed. It identifies the "
        "hold as yours rather than whichever one the server currently reports.")),
) -> None:
    """Let an agent take work again, whatever stopped it.

    Two different things stop an agent and they are released differently: a
    lifecycle rule pausing it after too many failures, and an operator holding it
    with `charter pause`. Which one it was is not the caller's problem — this is
    the verb for "run again" either way.
    """
    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant, instance)
            if wf is None:
                _err(f"{agent!r} has not been applied yet")
                raise typer.Exit(1)
            wf = await cp.get_workflow(wf.id)
            state = wf.workflow_state.value

            # An operator hold first: it outranks whatever the lifecycle policy
            # thinks, and releasing it restores the state the policy calls for
            # anyway — so activating underneath one would be both wrong and
            # pointless.
            if wf.suspension is not None:
                if wf.suspension.finalized_at is None:
                    # Two very different reasons it hasn't finalized, and only one
                    # of them resolves by waiting. A task parked on a human never
                    # drains on its own, so "try again in a moment" would be advice
                    # that can't come true.
                    if wf.lifecycle_state.value in ("awaiting_approval",
                                                    "awaiting_input"):
                        _warn(f"{agent} is paused, but a task is still waiting on "
                              f"you — that has to be answered before the hold can "
                              f"be released")
                        ui.detail(f"charter pending {agent} --instance {short(wf.id)}")
                    else:
                        _warn(f"{agent} is still finishing the task it was on")
                        ui.detail("the hold is recorded; it can be released once "
                                  "that run drains — try again in a moment")
                    raise typer.Exit(1)
                # Releasing whatever hold happens to be there is how you silently
                # undo someone else's pause: they stopped the agent for a reason you
                # cannot see, and afterwards nothing records that their hold existed.
                # BoundFlow returns an id per hold precisely so that is answerable,
                # so name the one you placed.
                if suspension and suspension != wf.suspension.suspension_id:
                    _err(f"{agent} is held by a different suspension "
                         f"({short(wf.suspension.suspension_id)}), not "
                         f"{short(suspension)}")
                    ui.detail(f"reason: {wf.suspension.reason or 'none given'}")
                    ui.detail("your hold was released and another placed since, "
                              "or this id belongs to a different agent")
                    raise typer.Exit(1)
                if not suspension:
                    _warn(f"{agent} is held, and you have not said which hold")
                    ui.detail(f"hold {wf.suspension.suspension_id}")
                    ui.detail(f"reason: {wf.suspension.reason or 'none given'}")
                    ui.detail(f"charter resume {agent} --suspension "
                              f"{wf.suspension.suspension_id}")
                    ui.detail("charter describe " + agent + "  # if you need the id")
                    raise typer.Exit(1)
                await cp.resume_workflow(wf.id, wf.suspension.suspension_id)
                held = wf.suspension.reason
                _ok(f"{ui.ref('agent', agent)} released"
                    + (f"  (held: {held})" if held else ""))
                ui.detail("anything queued while it was paused runs now")
                return

            if state == "active":
                ui.dim(f"{agent} is already active")
                return

            # A platform interruption disables the workflow; that needs resolving
            # before it can be activated at all.
            if wf.last_interrupted_request_id:
                # This clears the interruption *and* re-activates — BoundFlow's own
                # `workflow resolve` says so. Falling through to activate_workflow
                # then failed the precondition, because by that point the workflow
                # was already active: the command did its job and reported a
                # traceback, which is the worst of both.
                await cp.resolve_interrupted_workflow(wf.id, wf.last_interrupted_request_id)
                _ok(f"{ui.ref('agent', agent)} active  (platform interruption cleared)")
                ui.detail(f"the interrupted task did not resume — charter run {agent}")
                return

            # Activating a paused or cooling-down workflow has to name the policy
            # decision that stopped it — proof you're resuming the thing you looked
            # at, not racing a rule that fired since. Empty is correct only for a
            # workflow that has never had a decision.
            await cp.activate_workflow(wf.id, wf.last_policy_decision_request_id)
            ui.ok(f"{ui.ref('agent', agent)} active  (was {state})")

    asyncio.run(go())


@app.command()
def delete(
    agent: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation"),
    tenant: str = TENANT,
    instance: str = INSTANCE,
) -> None:
    """Delete an agent's workflow and its history. Mostly for iterating locally."""
    if not yes:
        typer.confirm(f"Delete {agent} and all its run history?", abort=True)

    async def go():
        async with _cp() as cp:
            wf = await _workflow_for(cp, agent, tenant, instance)
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
