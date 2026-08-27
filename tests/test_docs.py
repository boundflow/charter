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
