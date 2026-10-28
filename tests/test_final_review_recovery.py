from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from charter.config.agent import AgentConfig
from charter.harness.delegation_prerequisites import DelegationPrerequisites


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, ident, **args):
    return AIMessage(content='', tool_calls=[{'name': name, 'id': ident, 'args': args}])


@pytest.mark.asyncio
async def test_real_specialist_return_and_premature_answer_resume_for_actual_parent_review():
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import InMemorySaver
    invoked = []

    @tool
    def review():
        """Execute independent acceptance; failing assertions still need reporting."""
        invoked.append('review')
        return {'ok': False, 'cases': ['unresolved assertion']}

    parent = ToolModel(responses=[call('review', 'baseline'),
        call('task', 'handoff', subagent_type='worker', description='Repair assigned source'),
        AIMessage(content='Premature answer'), call('review', 'current'),
        AIMessage(content='Current parent review still reports an unresolved assertion')])
    child = ToolModel(responses=[AIMessage(content='Specialist reports passing checks')])
    agent = create_deep_agent(model=parent, tools=[review],
        middleware=[DelegationPrerequisites([], ['review'], final_review_retries=2)],
        subagents=[{'name': 'worker', 'description': 'Repair source',
                    'system_prompt': 'Return findings', 'model': child, 'tools': []}],
        checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'actual-handoff'}}
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Build'}]}, config)
    assert invoked == ['review', 'review']
    assert result['messages'][-1].content == 'Current parent review still reports an unresolved assertion'
    retained = (await agent.aget_state(config)).values
    assert retained['final_review_recovery_attempts'] == ['handoff']
    assert 'final_review_recovery_attempts' not in agent.get_input_jsonschema()['properties']
    assert 'final_review_recovery_attempts' not in result
    records = retained['delegation_receipts']
    assert any(r['tool'] == 'review' and r['tool_call_id'] == 'current'
               and r['after_task_id'] == 'handoff' for r in records)


def test_retry_budget_survives_compaction_and_cannot_accept_repeated_provisional_answers():
    from tests.test_delegation_prerequisites import compacting_middleware
    from langgraph.checkpoint.memory import InMemorySaver

    @tool
    def review():
        """Review current source."""
        return 'actual check failed'

    @tool
    def task():
        """Execute a specialist."""
        return 'changed source'

    model = ToolModel(responses=[call('review', 'baseline'), call('task', 'handoff'),
        AIMessage(content='Premature 1'), AIMessage(content='Premature 2'),
        AIMessage(content='Premature 3')])
    checkpoint = InMemorySaver()
    agent = create_agent(model, tools=[review, task],
        middleware=[compacting_middleware(),
                    DelegationPrerequisites([], ['review'], final_review_retries=2)],
        checkpointer=checkpoint)
    config = {'configurable': {'thread_id': 'bounded'}}
    with pytest.raises(ValueError, match='without fresh review: review'):
        agent.invoke({'messages': [{'role': 'user', 'content': 'Build'}]}, config)
    assert agent.get_state(config).values['final_review_recovery_attempts'] == ['handoff', 'handoff']


def test_failed_review_does_not_unlock_finalization_and_new_handoff_has_own_retry_budget():
    guard = DelegationPrerequisites([], ['review'], final_review_retries=1)
    state = {'messages': [], 'delegation_receipts': [
        {'tool': 'task', 'tool_call_id': 'first', 'returned': True}],
        'final_review_recovery_attempts': ['first']}
    with pytest.raises(ValueError, match='without fresh review'):
        guard.after_agent(state, None)
    state['delegation_receipts'].append({'tool': 'task', 'tool_call_id': 'second', 'returned': True})
    update = guard.after_agent(state, None)
    assert update['final_review_recovery_attempts'] == ['second']
    state['final_review_recovery_attempts'] += update['final_review_recovery_attempts']
    state['messages'] = [call('review', 'bad'),
        ToolMessage(content='Error executing tool review: infrastructure unavailable',
                    status='error', tool_call_id='bad')]
    with pytest.raises(ValueError, match='without fresh review'):
        guard.after_agent(state, None)


def test_configuration_and_coordinator_only_wiring():
    from charter.config.loader import load_agent
    from charter.workflows.loop import Loop
    bundle = load_agent(Path(__file__).parents[1] / 'examples/refund-triage')
    raw = bundle.latest.model_dump(by_alias=True)
    for value in [-1, 3, True]:
        with pytest.raises((ValidationError, ValueError)):
            AgentConfig.model_validate({**raw, 'coordinator_final_review_retries': value})
    with pytest.raises(ValidationError, match='requires review tools and configured specialists'):
        AgentConfig.model_validate({**raw, 'coordinator_final_review_retries': 1})
    raw.update(coordinator_final_review_retries=2, subagent_review_tools=['support__create_refund'],
        subagents=[{'name': 'worker', 'description': 'Repair', 'prompt': 'Inspect source'}])
    cfg = AgentConfig.model_validate(raw)
    loop = Loop(cfg, bundle.runtime, tools=SimpleNamespace(langchain_tools=lambda: []),
        chat_model=lambda _: object(), store_url='postgresql://unused')
    wiring = loop._wiring(SimpleNamespace(wiring={'subagents': [{'name': 'worker'}]}), None, [])
    guard = next(m for m in wiring['middleware'] if isinstance(m, DelegationPrerequisites))
    assert guard.final_review_retries == 2
    assert not any(isinstance(m, DelegationPrerequisites) for m in wiring['subagents'][0].get('middleware', []))
