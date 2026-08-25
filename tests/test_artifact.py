"""Sealing an agent version.

A registry is not needed to test what matters here — that the bytes are the right
bytes, and the same every time. The push path is exercised against a local
registry in the e2e suite instead.
"""

from __future__ import annotations

import gzip
import io
import subprocess
import sys
import tarfile
from pathlib import Path

from charter.artifact import pack
from charter.config.loader import load_agent

DEMO = Path(__file__).parent.parent / "demo" / "leads" / "leads-finder"


def bundle():
    return load_agent(DEMO)


def test_behaviour_travels_and_policy_does_not():
    """The split the whole design rests on. Policy converges on one control plane
    and stays mutable; sealing it here would make a thing you can change look like
    a thing you cannot — and would mean tightening a budget needed a new version."""
    packed = pack(bundle(), 1)

    assert "v1.yaml" in packed.files
    assert any(f.endswith("SKILL.md") for f in packed.files), "skills travel"
    assert not any("runtime.yaml" in f for f in packed.files)
    assert not any("lifecycle.yaml" in f for f in packed.files)


def test_the_same_input_packs_to_the_same_digest():
    """Across processes, not just within one.

    `tarfile.open(mode="w:gz")` stamps the current time into the gzip header, so
    two packs of identical bytes a second apart differ — which reads exactly like
    a content change, and would make every push look like a new version.
    """
    code = (
        "from charter.artifact import pack;"
        "from charter.config.loader import load_agent;"
        "from pathlib import Path;"
        f"print(pack(load_agent(Path({str(DEMO)!r})), 1).digest)"
    )
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=DEMO.parent.parent.parent).stdout.strip()
            for _ in range(2)}

    assert len(runs) == 1, f"digest varies between processes: {runs}"


def test_changing_a_skill_changes_the_digest():
    """The property that makes a version mean something. If a skill could change
    under a fixed version, a rollback would restore config the agent no longer has
    the knowledge to run."""
    before = pack(bundle(), 1).digest
    skill = next(p for p in (DEMO / "v1" / "skills").rglob("SKILL.md"))
    original = skill.read_text()
    try:
        skill.write_text(original + "\n<!-- edited -->\n")
        after = pack(load_agent(DEMO), 1).digest
    finally:
        skill.write_text(original)

    assert before != after
    assert pack(load_agent(DEMO), 1).digest == before, "and restoring restores it"


def test_it_is_an_ordinary_gzipped_tarball():
    """No private format. The same shape a Helm chart or an OPA bundle takes, so
    `oras pull` and anything else that speaks OCI can read it."""
    packed = pack(bundle(), 1)

    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(packed.tar))) as tar:
        assert "v1.yaml" in tar.getnames()


def test_the_tag_comes_from_the_config_not_the_caller():
    """An artifact tagged v2 whose config says version 1 would break the one thing
    the design rests on — that a worker reads an artifact and knows which handler
    to register."""
    packed = pack(bundle(), 1)

    assert packed.tag == "v1"
    assert packed.reference("ghcr.io/acme/agents") == \
        "ghcr.io/acme/agents/leads-finder:v1"
