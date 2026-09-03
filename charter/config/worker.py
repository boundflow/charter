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
    # a way a uuid isn't. Required, not defaulted: a workflow's tenant is fixed at
    # creation, so an agent that silently landed in the wrong one can only be
    # deleted and recreated, losing its run history and every lifecycle window.
    # Must already exist — `charter tenant create`.
    tenant: str = Field(min_length=1)
    # Pins an exact tenant by id, for when a name would be ambiguous. Wins over `tenant`.
    tenant_id: str = ""
    # Where workers claim operations. A different address from `endpoint`, which is
    # the control API the CLI reads: BoundFlow serves them on separate ports, and a
    # hosted control plane puts them on separate hosts. Unset falls back to
    # BOUNDFLOW_WORKER_ADDRESS, then to BoundFlow's own default.
    worker_endpoint: str = ""


class Llm(Base):
    """Inference is bring-your-own: this key lives in the operator's environment,
    and the control plane never sees it or the traffic.

    The model *name* isn't here — it's in each agent's versioned config, so a model
    change is a version bump you can roll back. This is only where the credential
    comes from and which provider integration to build.
    """

    provider: str = Field(default="anthropic", min_length=1, description=(
        "Any provider LangChain can build: anthropic, openai, google_genai, "
        "bedrock, ollama, huggingface, groq, mistralai, xai. The list is theirs, "
        "so it is not enumerated here — an unknown one fails at boot, naming it."))
    # Optional: a local runtime needs no key, and some providers take credentials
    # from the environment (bedrock from AWS's chain, vertex from ADC).
    api_key: str = ""
    # For an OpenAI-compatible server you host — vLLM, LM Studio, llama.cpp — or a
    # provider on a non-default host.
    base_url: str = ""


class Store(Base):
    """Where the harness keeps state between rounds.

    The agent's conversation and its files have to outlive the operation, or every
    gate would start the task over. Both live here — a checkpointer for the thread
    and a store for the filesystem — and both are the harness's own Postgres
    integrations, so Charter provisions the connection and nothing else.

    It is deliberately the operator's database, not the control plane's. What
    accumulates in it is prompts, reasoning, tool output and whatever customer data
    the agent touched; a governance product has no business holding that. Pointing
    it at the Postgres you already run for BoundFlow is one connection string and
    no new infrastructure — but it stays yours.
    """

    url: str = Field(min_length=1, description=(
        "postgresql://... — or $ENV_VAR, resolved at boot like every other secret."))


class Served(Base):
    """One agent this worker serves, from a directory or from a registry.

    Two spellings, because they answer different questions. A directory is what you
    want while writing an agent — edit the yaml, restart, see it. A registry ref is
    what you want once it is real: the artifact is immutable, the same bytes reach
    every worker, and nothing has to be checked out.

        serves:
          - agent: leads-finder          # from ./leads-finder
            versions: [1]
          - agent: leads-finder          # from ghcr.io/acme/agents/leads-finder
            versions: [1, 2]
            repository: ghcr.io/acme/agents
          - ref: ghcr.io/acme/agents/leads-finder:v1

    `versions` is a list, not a single number, because a `set_version` rollback
    dispatches operations at the old version and this process must still be able to
    build that agent. Dropping a version a lifecycle rule can target strands the
    control plane.

    `repository` derives one address per version, so a registry-served agent takes
    the same `versions` list a directory does. `ref` names one artifact and cannot:
    serving two versions that way means two entries, and nothing checks you wrote
    the second.
    """

    agent: str | None = Field(default=None, pattern=AGENT_NAME)
    versions: list[int] = Field(default_factory=list)
    insecure: bool = Field(default=False, description=(
        "Plain HTTP to the registry. For a local one; a real registry is TLS."))
    ref: str | None = Field(default=None, description=(
        "An OCI reference like ghcr.io/acme/agents/leads-finder:v1. The artifact "
        "names its own agent and version, so neither is repeated here."))
    repository: str | None = Field(default=None, description=(
        "Where this agent's artifacts live, like ghcr.io/acme/agents. Each served "
        "version is pulled from `<repository>/<agent>:v<N>` — the same address "
        "`charter push` writes to."))

    @property
    def from_registry(self) -> bool:
        return self.ref is not None or self.repository is not None

    @model_validator(mode="after")
    def _check(self) -> Served:
        if bool(self.ref) == bool(self.agent):
            raise ValueError(
                "give either `agent` (a directory) or `ref` (a registry artifact), "
                "not both — an artifact names the agent it holds")
        if self.ref and self.repository:
            raise ValueError(
                "`ref` and `repository` both address the artifact: `ref` names one "
                "version, `repository` derives every version from the agent name")
        if self.repository:
            # Only the last segment is checked: a registry may carry a port,
            # localhost:5000/acme.
            if ":" in self.repository.rsplit("/", 1)[-1]:
                raise ValueError(
                    f"{self.repository!r} carries a tag — `repository` is the path "
                    f"agents live under, and the version is appended per entry")
        if self.ref:
            if self.versions:
                raise ValueError(
                    "`versions` doesn't apply to a `ref`: an artifact is one "
                    "version, and it says which inside")
            if ":" not in self.ref.rsplit("/", 1)[-1]:
                raise ValueError(
                    f"{self.ref!r} has no tag — pin the version you mean, "
                    f"e.g. {self.ref}:v1")
            return self
        if not self.versions:
            raise ValueError("`versions` is required when serving from a directory")
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
    # All three are the same signed POST to a URL you supply; they differ only in
    # the body shape, because chat products reject arbitrary JSON.
    #   webhook   Charter's own envelope
    #   slack     {"text": ...}
    #   telegram  {"chat_id": ..., "text": ...}
    kind: Literal["webhook", "slack", "telegram"]
    url: str = Field(min_length=1)
    # Body is signed HMAC-SHA256 into X-Charter-Signature. Omitting it sends
    # unsigned, which is only sane for localhost.
    secret: str | None = None
    timeout_seconds: int = Field(default=5, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    # telegram only: which chat to send to.
    chat_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Channel:
        if self.kind == "telegram" and not self.chat_id:
            raise ValueError("kind `telegram` requires `chat_id`")
        return self


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
    store: Store
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
