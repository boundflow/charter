"""The YAML in the documentation is real configuration, not illustration.

DESIGN.md's two flagship examples had both stopped validating: `mcp[].tools` was
shown as a `{read: [...], act: [...]}` mapping after it became a list, the agent
file carried a top-level `outcome:` block that no longer exists, and the runtime
file set `per_run.max_iterations`, which was removed. Nothing failed, because
nothing reads the docs.
"""
import re
from pathlib import Path

import pytest
import yaml

from charter.config.agent import AgentConfig
from charter.config.runtime import RuntimePolicyFile
from charter.config.worker import WorkerManifest

ROOT = Path(__file__).parent.parent
DOCS = ["README.md", "DESIGN.md", "demo/leads/README.md", "deploy/README.md"]


def _model_for(doc: dict):
    """Which config a block is, or None when it is a fragment rather than a file.

    Fragments are the common case — a doc shows three keys to make a point — so
    identification is by the marker keys a whole file must have.
    """
    if doc.get("kind") == "AgentConfig":
        return AgentConfig
    if doc.get("kind") == "RuntimePolicy":
        return RuntimePolicyFile
    if doc.get("kind") == "Worker":
        return WorkerManifest
    return None


def _blocks():
    for name in DOCS:
        path = ROOT / name
        if not path.exists():
            continue
        for i, block in enumerate(re.findall(r"```ya?ml\n(.*?)```", path.read_text(), re.S)):
            yield name, i, block


@pytest.mark.parametrize("name,index,block", list(_blocks()),
                         ids=lambda v: str(v)[:20] if isinstance(v, str) else str(v))
def test_documented_yaml_is_valid(name, index, block):
    try:
        doc = yaml.safe_load(block)
    except yaml.YAMLError as e:
        pytest.fail(f"{name} block {index} is not YAML: {e}")
    if not isinstance(doc, dict):
        return
    model = _model_for(doc)
    if model is None:
        return
    model.model_validate(doc)


def test_readme_quickstart_is_what_init_writes():
    """The quickstart YAML the README prints is byte-for-byte what `charter init`
    produces, so following the prose and running the command give the same files.

    Drift here is silent and lands on a first-time reader: they copy the README,
    the scaffold wrote something else, and the error names a file they never typed.
    """
    from charter import scaffold

    readme = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```ya?ml\n(.*?)```", readme, re.S)
    for body in scaffold.files("triage").values():
        assert body in blocks, (
            "README's quickstart no longer matches `charter init` — update whichever "
            f"of the two is wrong. Missing:\n{body}")


def test_init_writes_configs_that_load(tmp_path):
    """`charter init` produces a project that `charter validate` accepts.

    A scaffold that needs editing before it parses is not a scaffold.
    """
    from charter import scaffold
    from charter.config.loader import load_project

    for rel, body in scaffold.files("triage").items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)

    project = load_project(tmp_path / "worker.yaml")
    assert set(project.agents) == {"triage"}
    assert set(project.agents["triage"].versions) == {1}


def test_init_adds_a_second_agent_to_an_existing_manifest(tmp_path):
    """A second `charter init` in the same project extends `serves:` rather than
    refusing, which is how someone adds their next agent.

    The manifest is edited as text, so comments and formatting a person wrote
    survive the edit.
    """
    from charter import scaffold
    from charter.config.loader import load_project

    for rel, body in scaffold.files("triage").items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)

    worker = tmp_path / "worker.yaml"
    worker.write_text(worker.read_text().replace(
        "serves:", "# the agents this process serves\nserves:"))

    worker.write_text(scaffold.add_to_serves(worker.read_text(), "escalations"))
    (tmp_path / "escalations").mkdir()
    (tmp_path / "escalations" / "v1.yaml").write_text(
        scaffold.files("escalations")["escalations/v1.yaml"])

    assert "# the agents this process serves" in worker.read_text()
    project = load_project(worker)
    assert set(project.agents) == {"triage", "escalations"}


def test_add_to_serves_keeps_trailing_content(tmp_path):
    """The entry lands inside `serves:`, not after whatever follows it."""
    from charter import scaffold

    text = scaffold.WORKER.format(name="triage") + "\nmodel_pricing:\n  x: { input: 1, output: 2 }\n"
    out = scaffold.add_to_serves(text, "escalations")
    assert out.index("- agent: escalations") < out.index("model_pricing:")
    assert out.endswith("  x: { input: 1, output: 2 }\n")
