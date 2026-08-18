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
from boundflow.anthropic_client import AnthropicLlmClient

from . import ui
from .config.loader import AgentBundle, Project
from .config.worker import Channel, WorkerManifest
from .memory import AuditMemory
from .mcp.client import QuarantineError, ToolSet
from .notify import Notifier
from .workflows.loop import OP_EXECUTE_ACT, Loop

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
    # Tests substitute a deterministic client here. Everything else in an
    # end-to-end run stays real, because the model is the one component where
    # determinism is worth more than fidelity.
    llm: object | None = None

    async def run(self) -> None:
        manifest = self.project.manifest
        cp = ControlPlaneClient(
            resolve(manifest.control_plane.endpoint),
            resolve(manifest.control_plane.api_key))

        async with cp:
            worker = BoundFlowWorker(
                llm=self.llm or _llm(manifest),
                api_key=resolve(manifest.control_plane.api_key),
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
            bundle = self.project.agents[spec.agent]
            for version in spec.versions:
                key = (spec.agent, version)
                cfg = bundle.versions[version]
                tools = ToolSet()
                # Memory is built per task from the operation's own workflow_id, so
                # the worker never has to look a workflow up by name.
                memory = AuditMemory(cp) if cfg.memory else None

                served = Served(bundle, version, tools,
                                Loop(cfg, bundle.runtime, tools, memory))
                try:
                    await tools.connect(cfg)
                except QuarantineError as e:
                    served.quarantined = str(e)
                    log.error("quarantined %s@v%d: %s", spec.agent, version, e)
                self.served[key] = served

    # ── registration ────────────────────────────────────────────────────────

    def _register(self, worker: BoundFlowWorker, notifier: Notifier) -> None:
        """Two handlers per (agent, version). No per-agent code — the difference
        between two Charter agents is entirely their config."""
        for (agent, version), served in self.served.items():
            worker.workflow(agent, version=version)(self._entry(served))
            worker.operation(agent, OP_EXECUTE_ACT)(self._execute_act(served))

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

    def _execute_act(self, served: Served):
        async def execute_act(ctx):
            return await served.loop.execute_act(ctx)

        return execute_act

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


def _llm(manifest: WorkerManifest):
    if manifest.llm.provider == "anthropic":
        return AnthropicLlmClient(api_key=resolve(manifest.llm.api_key))
    raise RuntimeError(
        f"llm provider {manifest.llm.provider!r} is declared but not wired up yet")


async def run_worker(project: Project, llm=None) -> None:
    worker = CharterWorker(project, llm=llm)
    try:
        await worker.run()
    finally:
        await worker.aclose()
