"""The Worker manifest — `worker.yaml`, one per worker process.

The worker itself is generic: a single implementation of the feedback loop serving
every Charter agent. This file is what makes a fleet manageable — which process
runs which agent at which version is declarative, so agents can be sharded across
workers, or a canary can hold only v2 while the fleet stays on v1.

Not versioned. Everything secret is an ${ENV_VAR} reference resolved at boot.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .agent import AGENT_NAME, Base

Event = Literal["approval_requested", "input_requested"]


class ControlPlane(Base):
    endpoint: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    # Which tenant these agents belong to, by name — legible in a committed file in
    # a way a uuid isn't. It must already exist (`charter tenant create`); Charter
    # never creates one as a side effect of applying an agent.
    tenant: str = "default"
    # Pins an exact tenant by id, for when a name would be ambiguous. Wins over `tenant`.
    tenant_id: str = ""


class Llm(Base):
    """Inference is bring-your-own: this key lives in the operator's environment,
    and the control plane never sees it or the traffic."""

    provider: Literal["anthropic", "langchain"]
    api_key: str = Field(min_length=1)


class Served(Base):
    """`versions` is a list, not a single number, because a `set_version` rollback
    dispatches operations at the old version and this process must still be able to
    build that agent from disk. Dropping a version a lifecycle rule can target
    strands the control plane."""

    agent: str = Field(pattern=AGENT_NAME)
    versions: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Served:
        if any(v < 1 for v in self.versions):
            raise ValueError("versions must be >= 1")
        if len(set(self.versions)) != len(self.versions):
            raise ValueError("duplicate version")
        return self


class Channel(Base):
    """One notification destination. A URL and an optional secret are the only
    credential material Charter holds — no OAuth, no vendor SDK, no integration to
    maintain."""

    name: str = Field(min_length=1)
    # `webhook` posts Charter's own JSON envelope. `slack` posts {"text": ...},
    # because Slack's incoming webhooks reject arbitrary JSON — a formatting
    # variant, not an integration: same signed POST to a URL you supply.
    kind: Literal["webhook", "slack"]
    url: str = Field(min_length=1)
    # Body is signed HMAC-SHA256 into X-Charter-Signature. Omitting it sends
    # unsigned, which is only sane for localhost.
    secret: str | None = None
    timeout_seconds: int = Field(default=5, gt=0)
    max_attempts: int = Field(default=3, ge=1)


class Route(Base):
    """Matched top to bottom, first match wins. Keyed by agent NAME — a string, not
    a path — so this file never reaches into an agent repo."""

    agent: str | None = None
    events: list[Event] | None = None
    channel: str = Field(min_length=1)


class Notifications(Base):
    channels: list[Channel] = Field(min_length=1)
    routes: list[Route] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Notifications:
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("duplicate channel name")
        known = set(names)
        for route in self.routes:
            if route.channel not in known:
                raise ValueError(
                    f"route targets channel {route.channel!r}, which is not defined")
        return self

    def resolve(self, agent: str, event: Event) -> Channel | None:
        """The channel an event for `agent` goes to, or None if nothing matches."""
        for route in self.routes:
            if route.agent is not None and route.agent != agent:
                continue
            if route.events is not None and event not in route.events:
                continue
            return next(c for c in self.channels if c.name == route.channel)
        return None


class TraceSink(Base):
    kind: Literal["none", "logging", "jsonl", "otel"]
    path: str | None = None
    endpoint: str | None = None

    @model_validator(mode="after")
    def _check(self) -> TraceSink:
        if self.kind == "jsonl" and not self.path:
            raise ValueError("trace_sink kind `jsonl` requires `path`")
        if self.kind == "otel" and not self.endpoint:
            raise ValueError("trace_sink kind `otel` requires `endpoint`")
        return self


class ModelPrice(Base):
    """Per 1M tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class WorkerManifest(Base):
    apiVersion: Literal["charter/v1"]
    kind: Literal["Worker"]

    name: str | None = None
    control_plane: ControlPlane
    llm: Llm
    # The ONE reference between the two artifacts, and it points one way: worker ->
    # agent, by name. Agent configs never reference a worker.
    agents_dir: str = "./agents"
    serves: list[Served] = Field(min_length=1)
    notifications: Notifications | None = None
    trace_sink: TraceSink | None = None
    # Tenant-global in BoundFlow, not per-agent — which is why it lives here.
    model_pricing: dict[str, ModelPrice] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> WorkerManifest:
        agents = [s.agent for s in self.serves]
        if len(set(agents)) != len(agents):
            raise ValueError("duplicate agent in `serves`")
        return self

    def served_versions(self, agent: str) -> set[int]:
        for s in self.serves:
            if s.agent == agent:
                return set(s.versions)
        return set()
