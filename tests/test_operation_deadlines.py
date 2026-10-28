from pathlib import Path

import pytest
from pydantic import ValidationError

from charter.config.runtime import RuntimePolicyFile
from charter.config.loader import load_agent
from charter.compile import compile_agent


def runtime(**options):
    return RuntimePolicyFile.model_validate({
        'apiVersion': 'charter/v1', 'kind': 'RuntimePolicy', 'agent': 'long-build',
        'per_run': {'max_llm_calls': 160, 'max_seconds': 7200},
        'limits': {'max_call_seconds': 600}, **options})


def test_long_round_requires_explicit_operator_ceiling():
    assert runtime().operation_timeout_seconds == 3600
    assert runtime(operation_timeout_ceiling_seconds=7200).operation_timeout_seconds == 7200
    assert runtime(operation_timeout_ceiling_seconds=10800).operation_timeout_seconds == 7200
    assert runtime(operation_timeout_ceiling_seconds=7200,
        per_run={'max_llm_calls': 2, 'max_seconds': 7200}).operation_timeout_seconds == 1200
    assert runtime(per_run={'max_llm_calls': 1, 'max_seconds': 1}).operation_timeout_seconds == 60


@pytest.mark.parametrize('seconds', [0, 59, 86401])
def test_operation_ceiling_remains_bounded(seconds):
    with pytest.raises(ValidationError):
        runtime(operation_timeout_ceiling_seconds=seconds)


def test_compiled_entry_operation_honors_explicit_long_budget():
    bundle = load_agent(Path(__file__).parents[1] / 'examples/refund-triage')
    bundle.runtime = runtime(agent=bundle.name, operation_timeout_ceiling_seconds=7200)
    compiled = compile_agent(bundle)
    assert compiled.workflow_config.invoke_timeout_seconds == 7200
    from charter.policy import OPERATION_TIMEOUT_SECONDS
    assert compiled.runtime_policy.custom[OPERATION_TIMEOUT_SECONDS] == 7200
