"""Any model LangChain can build, not only Anthropic."""
import shutil
from pathlib import Path

import pytest
import yaml

from charter.config.loader import load_project
from charter.config.worker import Llm
from charter.worker import CharterWorker

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def project(tmp_path):
    dst = tmp_path / "project"
    shutil.copytree(EXAMPLES, dst)
    raw = yaml.safe_load((dst / "worker.yaml").read_text())
    raw["store"] = {"url": "postgres://x/y"}
    (dst / "worker.yaml").write_text(yaml.safe_dump(raw))
    return load_project(dst / "worker.yaml")


def _worker(project, **llm):
    raw = dict(provider="anthropic", api_key="k")
    raw.update(llm)
    project.manifest.llm = Llm(**raw)
    return CharterWorker(project)


def test_a_provider_is_not_enumerated_by_charter():
    """The list belongs to LangChain: a provider it gains works here untouched."""
    assert Llm(provider="openai", api_key="k").provider == "openai"
    assert Llm(provider="ollama").provider == "ollama"


def test_a_local_runtime_needs_no_key():
    """Ollama, vLLM and friends authenticate nothing; requiring a key would mean
    inventing one."""
    assert Llm(provider="ollama").api_key == ""


def test_a_base_url_reaches_a_server_you_host():
    assert Llm(provider="openai", base_url="http://localhost:8000/v1").base_url


def test_an_unbuildable_provider_says_what_to_install(project):
    """The failure a customer actually hits is a missing integration package."""
    worker = _worker(project, provider="not-a-provider")
    with pytest.raises(RuntimeError, match="could not build"):
        worker._chat_model("some-model")


def test_anthropic_still_builds(project):
    worker = _worker(project, provider="anthropic", api_key="k")
    assert type(worker._chat_model("claude-haiku-4-5")).__name__ == "ChatAnthropic"
