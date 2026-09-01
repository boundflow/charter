"""The generic Charter worker.

One process, one implementation of the feedback loop, serving every agent its
manifest lists. What makes a fleet manageable is that this file has no per-agent
code in it: `worker.yaml` says which `(agent, version)` pairs to register, and each
one gets the same two handlers built from its config.

Quarantine is the other half. A worker serves several agents, so one broken MCP
server must not take the process down — that agent alone is marked unhealthy and
its tasks fail fast with the reason, which increments num_failures and lets its own
lifecycle rule pause it. The rest keep running.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field

from boundflow import BoundFlowWorker, ControlPlaneClient
from boundflow.langchain_client import LangChainLlmClient

from . import trace as charter_trace, ui
from .config.loader import AgentBundle, Project
from .config.worker import Channel, WorkerManifest
from .mcp.client import QuarantineError, ToolSet
from .notify import Notifier
from .provisioning.apply import resolve_tenant
from .workflows.loop import Loop
from .workflows.spawning import Spawner

log = logging.getLogger(__name__)

ENV_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

# How long to wait before retrying a quarantined agent's MCP connections. Long
# enough not to hammer a broken server, short enough that fixing credentials
# doesn't need a deploy.
REQUARANTINE_SECONDS = 60


def resolve(value: str) -> str:
    """Expand ${VAR} from the environment. Secrets live there, never in a file."""
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in os.environ:
            missing.append(name)
            return ""
        return os.environ[name]

    out = ENV_REF.sub(sub, value)
    if missing:
        raise RuntimeError(f"environment variable(s) not set: {', '.join(missing)}")
    return out


@dataclass
class Served:
    """One (agent, version) this worker runs."""

    bundle: AgentBundle
    version: int
    tools: ToolSet
    loop: Loop
    quarantined: str | None = None


@dataclass
class CharterWorker:
    project: Project
    served: dict[tuple[str, int], Served] = field(default_factory=dict)
    # Tests substitute a deterministic model here. Everything else in an
    # end-to-end run stays real, because the model is the one component where
    # determinism is worth more than fidelity. A factory rather than an instance:
    # the model name comes from each agent's versioned config, so one worker
    # serving three agents builds three models.
    chat_model: object | None = None
    # Where pulled artifacts are unpacked. A temp directory per process, because a
    # restart is a deploy and a cache would be a second answer to "what is v1"
    # sitting on a disk nobody looks at.
    _pulled_dir: object | None = None

    @property
    def _pulled(self):
        import tempfile
        from pathlib import Path
        if self._pulled_dir is None:
            self._pulled_dir = tempfile.TemporaryDirectory(prefix="charter-agents-")
        return Path(self._pulled_dir.name)

    def _chat_model(self, model: str):
        """The LangChain model for one agent, built from the operator's credential.

        Governed by the time the harness sees it — `run_governed` wraps this in a
        GovernedChatModel, which is what makes subagent calls count against the
        same budget as the parent's.
        """
        if self.chat_model is not None:
            return self.chat_model(model)
        from langchain_anthropic import ChatAnthropic

        manifest = self.project.manifest
        if manifest.llm.provider != "anthropic":
            raise RuntimeError(
                f"llm provider {manifest.llm.provider!r} is declared but not wired up")
        return ChatAnthropic(model=model, api_key=resolve(manifest.llm.api_key))

    async def run(self) -> None:
        manifest = self.project.manifest
        cp = ControlPlaneClient(
            resolve(manifest.control_plane.endpoint),
            resolve(manifest.control_plane.api_key))

        async with cp:
            # Resolved once, up front: every child a spawner creates belongs to the
            # same tenant as its parent, and looking it up per spawn would be a
            # round trip on the model's critical path.
            self._tenant_id = (manifest.control_plane.tenant_id
                               or await resolve_tenant(cp, manifest.control_plane.tenant))
            worker = BoundFlowWorker(
                # Workers claim operations on a different address from the control
                # API. Unset here, BoundFlow reads its own env var and default.
                address=resolve(manifest.control_plane.worker_endpoint) or None,
                # The harness drives the loop and gets its model per call via
                # run_governed, so nothing here uses BoundFlow's own client. It's
                # still required, so it's bridged from the same credential rather
                # than becoming a second place a key could come from.
                llm=LangChainLlmClient(lambda name: self._chat_model(name)),
                api_key=resolve(manifest.control_plane.api_key),
                # Traces carry prompts and tool arguments, so they go where the
                # manifest points and never to the control plane.
                trace_sink=charter_trace.build(
                    manifest.trace_sink, manifest.name or "charter"),
            )
            notifier = Notifier(manifest.notifications)

            await self._connect_all(cp)
            self._register(worker, notifier)

            self._banner()
            await worker.run()

    def _banner(self) -> None:
        rows = []
        for (agent, version), served in sorted(self.served.items()):
            cfg = served.bundle.versions[version]
            rows.append((agent, version, len(cfg.all_tools), len(cfg.gated_tools),
                         served.quarantined or "ready"))
        ui.worker_banner(self.project.manifest.name or "worker",
                         self.project.manifest.control_plane.tenant, rows)

    # ── boot ────────────────────────────────────────────────────────────────

    async def _connect_all(self, cp: ControlPlaneClient) -> None:
        """Connect each served version's MCP servers. A failure quarantines that
        agent rather than the process — the others are unaffected."""
        for spec in self.project.manifest.serves:
            bundle, versions = await self._bundle_for(cp, spec)
            for version in versions:
                key = (bundle.name, version)
                cfg = bundle.versions[version]
                allowed = bundle.runtime.authority.allowed_spawns
                missing = [a for a in allowed if a not in self.project.agents]
                tools = ToolSet().with_timeout(
                    bundle.runtime.limits.max_tool_seconds)
                served = Served(bundle, version, tools,
                                Loop(cfg, bundle.runtime, tools,
                                     self._chat_model,
                                     resolve(self.project.manifest.store.url),
                                     skills=bundle.skills.get(version),
                                     spawner=self._spawner(cp, allowed)))
                try:
                    if missing:
                        # Caught at boot rather than the first time the model tries
                        # it — a typo here is a config error, and discovering it as
                        # a tool result mid-run is far too late to be useful.
                        raise QuarantineError(
                            f"spawns names {', '.join(missing)}, which this project "
                            f"has no config for")
                    await tools.connect(cfg)
                    for server in tools.servers.values():
                        for tool, why in server.tightened.items():
                            log.info("gated by policy: %s (%s)",
                                     server.spec.qualified(tool), why)
                except QuarantineError as e:
                    served.quarantined = str(e)
                    log.error("quarantined %s@v%d: %s", bundle.name, version, e)
                self.served[key] = served

    async def _bundle_for(self, cp: ControlPlaneClient, spec):
        """The agent this entry names, and which of its versions to serve.

        A directory is read straight off disk. A registry ref is pulled, unpacked
        and then read the same way — `load_agent` cannot tell the difference, which
        is the whole point of the artifact being a plain tarball of the layout it
        already expects.

        An artifact carries no runtime.yaml, because that is policy and policy is
        applied rather than shipped. Its numbers come back from the control plane
        instead, so a worker serving an artifact enforces exactly what a worker
        serving a checkout does.
        """
        if not spec.from_registry:
            bundle = self.project.agents[spec.agent]
            return bundle, spec.versions

        from . import artifact, policy as charter_policy
        from .config.loader import load_agent

        if spec.repository:
            # Every version lands in one tree, which is the layout `load_agent`
            # already reads: v1.yaml beside v2.yaml, each with its own skills.
            for version in sorted(spec.versions):
                ref = artifact.ref_for(spec.repository, spec.agent, version)
                directory = artifact.pull(ref, self._pulled, insecure=spec.insecure)
                log.info("pulled %s v%d from %s", spec.agent, version, ref)
            bundle = load_agent(directory)
            versions = sorted(spec.versions)
            absent = [v for v in versions if v not in bundle.versions]
            if absent:
                # The tag promised a version the artifact doesn't hold. Fatal like
                # a failed pull, not a quarantine: nothing was served to isolate.
                raise ValueError(
                    f"{spec.repository}/{spec.agent} has no "
                    f"{', '.join(f'v{v}' for v in absent)} — the artifact holds "
                    f"{', '.join(f'v{v}' for v in sorted(bundle.versions))}")
        else:
            directory = artifact.pull(spec.ref, self._pulled, insecure=spec.insecure)
            bundle = load_agent(directory)
            versions = [max(bundle.versions)]
            log.info("pulled %s v%d from %s", bundle.name, versions[0], spec.ref)

        wf = await self._workflow_for(cp, bundle.name)
        if wf is not None:
            live = await cp.get_agent_runtime_policy(wf.id, bundle.name)
            bundle.runtime = charter_policy.runtime_file(bundle.name, live)
        else:
            # Nothing applied yet, so there is no policy to read. The defaults are
            # the ones runtime.yaml would have given, and `charter apply` replaces
            # them on the next boot.
            log.warning("%s: no instance on the control plane yet — running on "
                        "default limits until one is applied", bundle.name)
        return bundle, versions

    async def _workflow_for(self, cp: ControlPlaneClient, agent: str):
        for w in await cp.list_workflows():
            if w.workflow_type == agent and w.tenant_id == self._tenant_id:
                return w
        return None

    def _spawner(self, cp: ControlPlaneClient, allowed: list[str]) -> Spawner | None:
        """None unless this version declares `spawns`, which is what keeps the
        async tools off the model's list for agents that shouldn't delegate."""
        if not allowed:
            return None
        return Spawner(cp, self._tenant_id, self.project.agents, allowed)

    # ── registration ────────────────────────────────────────────────────────

    def _register(self, worker: BoundFlowWorker, notifier: Notifier) -> None:
        """One handler per (agent, version). No per-agent code — the difference
        between two Charter agents is entirely their config, and the difference
        between a fresh task and a resumed one is entirely in the harness's
        checkpoint."""
        for (agent, version), served in self.served.items():
            worker.workflow(agent, version=version)(self._entry(served))

        @worker.on_approval_requested
        async def _approval(request):
            await notifier.approval_requested(request)

        @worker.on_input_requested
        async def _input(request):
            await notifier.input_requested(request)

    def _entry(self, served: Served):
        async def entry(ctx):
            if quarantined := await self._recheck(served):
                # Fails the task with the reason, which trips num_failures and lets
                # the agent's own lifecycle rule pause it — a broken agent stops
                # accepting work without anyone paging an operator.
                ctx.mark_failed()
                from boundflow import Complete
                return Complete(result={"failed": True, "reason": quarantined})
            return await served.loop.entry(ctx)

        return entry

    async def _recheck(self, served: Served) -> str | None:
        """Retry a quarantined agent's connections in the background of a task
        arriving, so fixing credentials doesn't need a redeploy."""
        if served.quarantined is None:
            return None
        try:
            await served.tools.connect(served.bundle.versions[served.version])
        except QuarantineError as e:
            served.quarantined = str(e)
            return served.quarantined
        log.info("recovered %s@v%d", served.bundle.name, served.version)
        served.quarantined = None
        return None

    async def aclose(self) -> None:
        for served in self.served.values():
            await served.tools.aclose()


async def run_worker(project: Project, chat_model=None) -> None:
    worker = CharterWorker(project, chat_model=chat_model)
    try:
        await worker.run()
    finally:
        await worker.aclose()
