"""Loading a Charter project off disk, and the rules that need more than one file.

Each model validates itself in isolation. What's left is the cross-file consistency
that actually prevents outages — a `set_version` target no worker serves, a tool
limit for a tool nothing declares, a filename that disagrees with the version
inside it.

Problems are collected and raised together: fixing config one error per run is
miserable, and these files are edited by people who aren't reading a stack trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from .agent import AgentConfig, template_refs
from .lifecycle import LifecyclePolicyFile
from .runtime import RuntimePolicyFile, default_runtime
from .worker import WorkerManifest

VERSION_FILE = re.compile(r"^v(\d+)\.yaml$")
RUNTIME_FILE = "runtime.yaml"
LIFECYCLE_FILE = "lifecycle.yaml"


class ConfigError(Exception):
    """One or more problems in a Charter project."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(f"  - {p}" for p in problems))


def _read(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError([f"{path}: invalid YAML: {e}"]) from e
    if not isinstance(raw, dict):
        raise ConfigError([f"{path}: expected a mapping at the top level"])
    return raw


def _parse(model, path: Path, problems: list[str]):
    """Parse `path` into `model`, appending readable problems instead of raising."""
    try:
        return model.model_validate(_read(path))
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            problems.append(f"{path.name}: {loc}: {err['msg']}")
        return None
    except ConfigError as e:
        problems.extend(e.problems)
        return None


@dataclass
class AgentBundle:
    """One agent directory: every version of its config, plus its two policy files.

    Every version is held, not just the newest, because a `set_version` rollback
    dispatches operations at an old version number and the worker must still be able
    to build that agent.
    """

    name: str
    path: Path
    versions: dict[int, AgentConfig]
    runtime: RuntimePolicyFile
    lifecycle: LifecyclePolicyFile | None
    # True when no runtime.yaml was present and defaults were substituted.
    runtime_defaulted: bool = False
    # version -> that version's skills directory, when it has one. Charter ships
    # the directory as-is and the harness's own loader reads it, so there is no
    # manifest here that could drift from what's on disk.
    skills: dict[int, Path] = field(default_factory=dict)

    @property
    def latest(self) -> AgentConfig:
        return self.versions[max(self.versions)]

    @property
    def declared_tools(self) -> set[str]:
        """Every tool any version declares. Policy files are unversioned, so a tool
        they reference need only exist in some version."""
        return {t for cfg in self.versions.values() for t in cfg.all_tools}


def load_agent(path: Path) -> AgentBundle:
    """Load one agent directory. Raises ConfigError listing every problem found."""
    problems: list[str] = []
    # Resolved so `.name` is a real directory name: Path(".").name is "", which
    # would otherwise reach pydantic as an agent named "".
    path = Path(path).resolve()

    if not path.is_dir():
        raise ConfigError([f"{path}: not a directory"])

    versions: dict[int, AgentConfig] = {}
    for file in sorted(path.iterdir()):
        m = VERSION_FILE.match(file.name)
        if not m:
            continue
        cfg = _parse(AgentConfig, file, problems)
        if cfg is None:
            continue
        n = int(m.group(1))
        if cfg.version != n:
            problems.append(
                f"{file.name}: declares version {cfg.version} — the filename says {n}")
        if cfg.name != path.name:
            problems.append(
                f"{file.name}: name {cfg.name!r} does not match directory {path.name!r}")
        versions[n] = cfg

    # Checked before anything derived from the directory name, so "this isn't an
    # agent directory" is what you're told rather than a consequence of it.
    if not versions:
        problems.append(
            f"{path.name or path}: not an agent directory — expected v1.yaml "
            f"(and optionally runtime.yaml, lifecycle.yaml)")
        raise ConfigError(problems)

    # Both policy files are optional. The smallest useful agent is one v1.yaml:
    # runtime falls back to a conservative default (an agent always has a ceiling,
    # but you shouldn't need a second file to get one), and lifecycle is fleet
    # management you add once you have a fleet.
    runtime_path = path / RUNTIME_FILE
    runtime = (_parse(RuntimePolicyFile, runtime_path, problems)
               if runtime_path.exists() else default_runtime(path.name))

    lifecycle_path = path / LIFECYCLE_FILE
    lifecycle = _parse(LifecyclePolicyFile, lifecycle_path, problems) if lifecycle_path.exists() else None

    if problems:
        raise ConfigError(problems)

    skills = {}
    for n in versions:
        directory = _skills_dir(path, n, problems)
        if directory is not None:
            skills[n] = directory
    if problems:
        raise ConfigError(problems)

    bundle = AgentBundle(path.name, path, versions, runtime, lifecycle,
                         runtime_defaulted=not runtime_path.exists(),
                         skills=skills)
    _check_agent(bundle, problems)
    if problems:
        raise ConfigError(problems)
    return bundle


def _skills_dir(path: Path, version: int, problems: list[str]) -> Path | None:
    """This version's skills, if it has any.

    They live in `v<N>/skills/` beside `v<N>.yaml`, which makes them versioned by
    the same rule the config is: you don't edit a version, you make a new one.
    Editing a skill in place would leave `set_version` restoring an agent that
    behaves differently from the one it rolled back to, silently.

    The layout is the harness's, not ours — one directory per skill, each holding
    a SKILL.md. An author's existing skills work unchanged, and Charter never
    parses them; it ships them and lets the harness's own loader read them.
    """
    directory = path / f"v{version}" / "skills"
    if not directory.is_dir():
        return None
    named = [d for d in sorted(directory.iterdir()) if d.is_dir()]
    if not named:
        problems.append(f"v{version}/skills/ exists but holds no skill directories")
        return None
    for skill in named:
        if not (skill / "SKILL.md").is_file():
            problems.append(
                f"v{version}/skills/{skill.name}/ has no SKILL.md — the harness "
                "identifies a skill by that file")
    return directory


def _check_agent(b: AgentBundle, problems: list[str]) -> None:
    if b.runtime.agent != b.name:
        problems.append(
            f"{RUNTIME_FILE}: agent {b.runtime.agent!r} does not match directory {b.name!r}")

    tools = b.declared_tools
    for limit in b.runtime.per_run.tool_call_limits:
        if limit.tool not in tools:
            problems.append(
                f"{RUNTIME_FILE}: tool_call_limits references {limit.tool!r}, which no "
                f"version of this agent declares")

    if b.lifecycle is None:
        return

    if b.lifecycle.agent != b.name:
        problems.append(
            f"{LIFECYCLE_FILE}: agent {b.lifecycle.agent!r} does not match directory {b.name!r}")

    for rule in b.lifecycle.rules:
        if rule.when.tool and rule.when.tool not in tools:
            problems.append(
                f"{LIFECYCLE_FILE}: rule references tool {rule.when.tool!r}, which no "
                f"version of this agent declares")

    for target in b.lifecycle.version_targets:
        if target not in b.versions:
            problems.append(
                f"{LIFECYCLE_FILE}: set_version target {target} has no v{target}.yaml — "
                f"a rollback would restore a version that doesn't exist")


@dataclass
class Project:
    """A worker manifest and every agent it serves."""

    manifest: WorkerManifest
    path: Path
    agents: dict[str, AgentBundle]


def load_project(worker_yaml: Path) -> Project:
    """Load a worker manifest and the agents it serves. Raises ConfigError listing
    every problem across every file."""
    problems: list[str] = []
    worker_yaml = Path(worker_yaml)

    manifest = _parse(WorkerManifest, worker_yaml, problems)
    if manifest is None:
        raise ConfigError(problems)

    root = (worker_yaml.parent / manifest.agents_dir).resolve()
    agents: dict[str, AgentBundle] = {}

    for served in manifest.serves:
        if served.from_registry:
            # Nothing to read: the artifact is fetched at boot and validated then,
            # because a config error inside one is the registry's problem to report
            # and not something a local parse could have caught.
            continue
        agent_dir = root / served.agent
        if not agent_dir.is_dir():
            problems.append(
                f"{worker_yaml.name}: serves {served.agent!r}, but {agent_dir} does not exist")
            continue
        try:
            agents[served.agent] = load_agent(agent_dir)
        except ConfigError as e:
            problems.extend(f"{served.agent}/{p}" for p in e.problems)

    for served in manifest.serves:
        if served.from_registry:
            continue
        bundle = agents.get(served.agent)
        if bundle is None:
            continue
        for v in served.versions:
            if v not in bundle.versions:
                problems.append(
                    f"{worker_yaml.name}: serves {served.agent} v{v}, but "
                    f"{served.agent}/v{v}.yaml does not exist")
        # The check that actually prevents an outage: a lifecycle rule can roll this
        # agent to a version, and if no worker holds it the control plane dispatches
        # operations nobody can handle.
        if bundle.lifecycle:
            for target in bundle.lifecycle.version_targets:
                if target not in served.versions:
                    problems.append(
                        f"{served.agent}: lifecycle can set_version to {target}, but this "
                        f"worker serves only {sorted(served.versions)} — a rollback would "
                        f"strand the agent")

    if problems:
        raise ConfigError(problems)
    return Project(manifest, worker_yaml, agents)
