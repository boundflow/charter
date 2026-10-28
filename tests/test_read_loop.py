from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.tools import tool

from charter.harness.read_loop import ReadLoopRecovery, RepeatedReadLoop


class Request(SimpleNamespace):
    def override(self, **kwargs):
        return Request(**{**vars(self), **kwargs})


def history(count=3, *, arguments=None, results=None):
    messages = [HumanMessage(content='Inspect then implement')]
    for i in range(count):
        messages.extend([
            AIMessage(content='', tool_calls=[{'id': str(i), 'name': 'read_record', 'args': (arguments or [{}] * count)[i]}]),
            ToolMessage(content=[{'type': 'text', 'text': (results or ['same'] * count)[i], 'id': f'random-{i}'}], tool_call_id=str(i))])
    return messages


def test_feedback_is_computed_from_real_results_without_changing_history():
    messages = history()
    result = ReadLoopRecovery(['read_record'])._request(Request(messages=messages))
    assert result.messages[:-1] == messages
    assert len(messages) == 7
    assert 'last 3 completed' in result.messages[-1].content
    assert 'All existing requirements and tool restrictions' in result.messages[-1].content
    assert result.messages[-2] is messages[-1]


@pytest.mark.parametrize('messages', [
    history(2), history(arguments=[{}, {}, {'page': 2}]), history(results=['same', 'same', 'changed']),
    history() + [HumanMessage(content='Read it again after my external change')],
])
def test_ordinary_reads_and_changed_evidence_are_not_a_loop(messages):
    request = Request(messages=messages)
    assert ReadLoopRecovery(['read_record'])._request(request) is request


def test_unconfigured_errors_or_unmatched_results_do_not_count_as_success():
    messages = history()
    request = Request(messages=messages)
    assert ReadLoopRecovery([])._request(request) is request
    messages[-1] = messages[-1].model_copy(update={'status': 'error'})
    assert ReadLoopRecovery(['read_record'])._request(request) is request
    messages[-1] = messages[-1].model_copy(update={'status': 'success', 'content': 'Error executing tool read_record: failed'})
    assert ReadLoopRecovery(['read_record'])._request(request) is request
    messages[-1] = messages[-1].model_copy(update={'status': 'success', 'tool_call_id': 'unmatched'})
    assert ReadLoopRecovery(['read_record'])._request(request) is request


def test_identical_failed_reads_receive_feedback_then_stop():
    messages = history(6, results=['Error executing tool read_record: deferred'] * 6)
    recovery = ReadLoopRecovery(['read_record'])
    result = recovery._request(Request(messages=messages[:7]))
    assert 'identical failed content' in result.messages[-1].content
    assert result.messages[:-1] == messages[:7]
    with pytest.raises(RepeatedReadLoop, match='6 consecutive identical failed reads'):
        recovery._request(Request(messages=messages))


def test_changed_failure_does_not_trigger_recovery():
    messages = history(results=['Error executing tool read_record: deferred',
                                'Error executing tool read_record: deferred',
                                'Error executing tool read_record: missing'])
    request = Request(messages=messages)
    assert ReadLoopRecovery(['read_record'])._request(request) is request


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def batch_history(count=3):
    messages = [HumanMessage(content='Read relevant chunks then implement')]
    for i in range(count):
        calls = [{'id': f'{i}-{page}', 'name': 'read_record', 'args': {'page': page}}
                 for page in (1, 4, 5)]
        if i % 2:
            calls.reverse()
        messages.append(AIMessage(content='', tool_calls=calls))
        messages.extend(ToolMessage(content=f"page {call['args']['page']}", tool_call_id=call['id'])
                        for call in reversed(calls))
    return messages


def test_repeated_batches_match_arguments_and_results_not_parallel_order():
    messages = batch_history()
    result = ReadLoopRecovery(['read_record'])._request(Request(messages=messages))
    assert result.messages[:-1] == messages
    assert 'last 3 completed read batches' in result.messages[-1].content
    with pytest.raises(RepeatedReadLoop, match='6 consecutive identical successful read batches'):
        ReadLoopRecovery(['read_record'])._request(Request(messages=batch_history(6)))


@pytest.mark.parametrize('change', ['changed_result', 'error', 'missing', 'duplicate', 'write'])
def test_batch_recovery_does_not_hide_changes_errors_or_mixed_actions(change):
    messages = batch_history()
    if change == 'changed_result':
        messages[-1] = messages[-1].model_copy(update={'content': 'new data'})
    elif change == 'error':
        messages[-1] = messages[-1].model_copy(update={'status': 'error'})
    elif change == 'missing':
        messages.pop()
    elif change == 'duplicate':
        messages[-1] = messages[-2]
    else:
        calls = [dict(c) for c in messages[-4].tool_calls]
        calls[0]['name'] = 'write_record'
        messages[-4] = messages[-4].model_copy(update={'tool_calls': calls})
    request = Request(messages=messages)
    assert ReadLoopRecovery(['read_record'])._request(request) is request


@pytest.mark.asyncio
async def test_actual_parallel_read_batches_stop_after_six_complete_executions():
    from deepagents import create_deep_agent
    calls = []

    @tool
    def read_record(page: int):
        """Read a numbered page."""
        calls.append(page)
        return f'page {page}'

    model = ToolModel(responses=[m for m in batch_history(7) if isinstance(m, AIMessage)])
    agent = create_deep_agent(model=model, tools=[read_record], middleware=[ReadLoopRecovery(['read_record'])])
    with pytest.raises(RepeatedReadLoop, match='6 consecutive identical successful read batches'):
        await agent.ainvoke({'messages': [HumanMessage(content='Inspect then implement')]})
    assert sorted(calls) == [1] * 6 + [4] * 6 + [5] * 6


@pytest.mark.asyncio
async def test_actual_agent_stops_after_six_unchanged_reads_and_retains_real_calls():
    from deepagents import create_deep_agent
    calls = []

    @tool
    def read_record():
        """Read current record."""
        calls.append('executed')
        return 'unchanged record'

    model = ToolModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'read_record', 'args': {}, 'id': str(i)}])
        for i in range(7)])
    agent = create_deep_agent(model=model, tools=[read_record], middleware=[ReadLoopRecovery(['read_record'])])
    with pytest.raises(RepeatedReadLoop, match='6 consecutive identical successful reads'):
        await agent.ainvoke({'messages': [HumanMessage(content='Inspect then implement')]})
    assert calls == ['executed'] * 6


@pytest.mark.asyncio
async def test_recovery_allows_a_new_action_without_synthesizing_it():
    from deepagents import create_deep_agent
    calls = []

    @tool
    def read_record():
        """Read current record."""
        calls.append('read')
        return 'unchanged'

    @tool
    def next_step():
        """Take the next requested step."""
        calls.append('next')
        return 'next step executed'

    model = ToolModel(responses=[
        *[AIMessage(content='', tool_calls=[{'name': 'read_record', 'args': {}, 'id': str(i)}]) for i in range(3)],
        AIMessage(content='', tool_calls=[{'name': 'next_step', 'args': {}, 'id': 'next'}]),
        AIMessage(content='Continued'),
    ])
    agent = create_deep_agent(model=model, tools=[read_record, next_step], middleware=[ReadLoopRecovery(['read_record'])])
    result = await agent.ainvoke({'messages': [HumanMessage(content='Inspect then take next step')]})
    assert calls == ['read', 'read', 'read', 'next']
    assert result['messages'][-1].content == 'Continued'
    assert len([m for m in result['messages'] if isinstance(m, HumanMessage)]) == 1
