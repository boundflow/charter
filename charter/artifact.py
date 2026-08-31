"""An agent version, sealed and pushed to a registry.

A version of an agent is its behaviour: the objective, the tools it may call, the
shape it answers in, and the skills it can read. That has to reach every worker
that might run it, including ones that don't exist yet — so it is packaged, not
applied. Policy goes the other way: it converges on one control plane and stays
mutable, which is why it travels as an API call and never as an artifact.

    package what travels; API what converges

The format is an OCI artifact holding one gzipped tarball, which is what Helm
charts and OPA bundles are. That buys the customer's existing registry, their
existing `docker login`, digests, replication and retention — none of it ours to
build. `oras pull` retrieves the same bytes, so nothing here is a private format.

The artifact is self-describing: the config inside names the agent and its
version, and the tag is derived from that rather than typed. A tag that disagreed
with the config would break the one property the whole design rests on — that a
worker can read an artifact and know which handler to register.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

# `+gzip` because that is what is on the wire: one gzipped tar. A reader that only
# knows OCI can still unpack it without knowing anything about Charter.
MEDIA_TYPE = "application/vnd.charter.agent.v1.tar+gzip"
ARTIFACT_TYPE = "application/vnd.charter.agent.v1"


@dataclass
class Packed:
    """A sealed version, before it goes anywhere."""

    agent: str
    version: int
    tar: bytes
    digest: str          # sha256 of the tarball, as the registry will address it
    files: list[str]     # what went in, for the operator to see

    @property
    def tag(self) -> str:
        return f"v{self.version}"

    def reference(self, repository: str) -> str:
        """`registry/repo/agent:v1` — the ref a worker is pointed at."""
        return ref_for(repository, self.agent, self.version)


def ref_for(repository: str, agent: str, version: int) -> str:
    """The address of one version: `registry/repo/agent:v<N>`. The same address
    `charter push` writes, so publishing and pulling agree by construction."""
    return f"{repository.rstrip('/')}/{agent}:v{version}"


def pack(bundle, version: int) -> Packed:
    """Seal one version's behaviour into a tarball.

    Exactly two things go in: `v<N>.yaml`, and `v<N>/skills/` if it exists. Not
    `runtime.yaml` and not `lifecycle.yaml` — those are policy, they are applied
    rather than shipped, and sealing them here would make a mutable thing look
    immutable.

    Deterministic on purpose: names sorted, timestamps and ownership zeroed. The
    same input has to produce the same digest, or every push looks like a change
    and `--if-changed` could never mean anything.
    """
    config = bundle.path / f"v{version}.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"{bundle.name} has no v{version}.yaml")

    members: list[tuple[str, Path]] = [(config.name, config)]
    skills = bundle.skills.get(version)
    if skills is not None:
        for path in sorted(skills.rglob("*")):
            if path.is_file():
                members.append((f"v{version}/skills/{path.relative_to(skills)}", path))

    raw = io.BytesIO()
    # Uncompressed here, gzipped below. `mode="w:gz"` stamps the current time into
    # the gzip header, so two packs of identical bytes a second apart produce
    # different digests — which reads exactly like a content change and would make
    # every push look like a new version.
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, path in sorted(members):
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))

    # mtime=0 in the gzip header for the same reason.
    zipped = io.BytesIO()
    with gzip.GzipFile(fileobj=zipped, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(raw.getvalue())
    blob = zipped.getvalue()

    return Packed(
        agent=bundle.name,
        version=version,
        tar=blob,
        digest="sha256:" + hashlib.sha256(blob).hexdigest(),
        files=[name for name, _ in sorted(members)],
    )


def push(packed: Packed, repository: str, *, insecure: bool = False) -> str:
    """Publish to the customer's registry. Returns the ref, digest included.

    `oras` rather than anything of ours: it already speaks every registry's
    dialect, and it reads `~/.docker/config.json`, so the credential is the one
    they already have. A worker pulling this is doing what a node pulling an image
    does, which is the point — on Kubernetes that is an `imagePullSecret` and
    nothing Charter-specific at all.
    """
    import tempfile

    import oras.client

    client = oras.client.OrasClient(insecure=insecure)
    target = packed.reference(repository)

    # oras-py uploads from disk, so the sealed bytes are written out and pushed.
    # Named for what it is: the file name becomes the layer's title annotation,
    # and `oras pull` will write it back under that name.
    with tempfile.TemporaryDirectory() as tmp:
        blob = Path(tmp) / f"{packed.agent}-v{packed.version}.tar.gz"
        blob.write_bytes(packed.tar)
        response = client.push(
            target=target,
            files=[f"{blob}:{MEDIA_TYPE}"],
            manifest_annotations={
                # Readable in any registry UI, and enough to identify an artifact
                # without unpacking it.
                "dev.charter.agent": packed.agent,
                "dev.charter.version": str(packed.version),
                "org.opencontainers.image.title": f"{packed.agent} v{packed.version}",
            },
            disable_path_validation=True,
        )

    # The registry addresses an artifact by its *manifest* digest. `packed.digest`
    # is the tarball's — the right identity for "has the content changed", and the
    # wrong one to put after an `@`, where it would simply fail to resolve.
    manifest = (response.headers or {}).get("Docker-Content-Digest", "")
    return f"{target}@{manifest}" if manifest else target


def pull(ref: str, into: Path, *, insecure: bool = False) -> Path:
    """Fetch an agent version from a registry and unpack it. Returns its directory.

    The unpacked tree is what `load_agent` already reads — `v<N>.yaml` beside a
    `v<N>/skills/` — so nothing downstream can tell whether a worker was given a
    checkout or an artifact. That is the point: one code path, two ways in.

    Nothing here is cached between boots. An artifact is small, a restart is a
    deploy, and a cache would be a second answer to "what is v1" living on a disk
    nobody looks at.
    """
    import tarfile

    import oras.client

    last = ref.rsplit("/", 1)[-1]
    agent, _, tag = last.partition(":")
    directory = into / agent
    directory.mkdir(parents=True, exist_ok=True)

    client = oras.client.OrasClient(insecure=insecure)
    # A blob directory per ref: several versions unpack into one tree, and a
    # shared one could hand back an earlier pull's layer.
    got = client.pull(target=ref, outdir=str(into / f".{agent}-{tag or 'latest'}-blob"))
    blobs = [Path(p) for p in got if Path(p).is_file()]
    if not blobs:
        raise ValueError(f"{ref} held nothing — is it a Charter agent artifact?")

    with tarfile.open(blobs[0]) as tar:
        _safe_extract(tar, directory)
    return directory


def _safe_extract(tar, into: Path) -> None:
    """Unpack, refusing anything that would land outside `into`.

    A tarball is a thing someone else built. `..` in a member name is the oldest
    trick there is, and the worker unpacking one runs wherever the customer's
    credentials are.
    """
    root = into.resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"artifact contains an escaping path: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"artifact contains a link: {member.name!r}")
    tar.extractall(root, filter="data")
