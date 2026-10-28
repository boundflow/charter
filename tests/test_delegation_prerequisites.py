from types import SimpleNamespace
import json

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from charter.harness.delegation_prerequisites import DelegationPrerequisites


def compacting_middleware():
    """Remove all completed tool history, as summarization can do."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import HumanMessage, RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    class Compact(AgentMiddleware):
        def before_model(self, state, runtime):
            if any(isinstance(m, ToolMessage) for m in state['messages']):
                return {'messages': [RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    HumanMessage(content='Summary: continue the task. No tool receipts retained.')]}

        async def abefore_model(self, state, runtime):
            return self.before_model(state, runtime)

    return Compact()


@pytest.mark.asyncio
async def test_checkpointed_receipts_survive_compaction_and_still_require_each_review():
    from langgraph.checkpoint.memory import InMemorySaver
    received, invoked = [], []

    @tool
    def reference():
        """Read reference."""
        invoked.append('reference')
        return 'contract'

    @tool
    def review():
        """Execute review."""
        invoked.append('review')
        return {'ok': False, 'expected': 42, 'actual': 1}

    @tool
    def task(description: str):
        """Delegate."""
        invoked.append('task')
        received.append(description)
        return 'changed'

    def delegate(ident):
        return AIMessage(content='', tool_calls=[{'name': 'task', 'id': ident,
                                                  'args': {'description': 'repair'}}])

    model = ToolModel(responses=[call('reference', 'ref'), call('review', 'baseline'),
        delegate('first'), delegate('must-review'), call('review', 'after-first'),
        delegate('second'), call('review', 'after-second'), AIMessage(content='done')])
    checkpoint = InMemorySaver()
    config = {'configurable': {'thread_id': 'one'}}

    def graph():
        return create_agent(model, tools=[reference, review, task],
            middleware=[compacting_middleware(), DelegationPrerequisites(['reference'], ['review'])],
            checkpointer=checkpoint, interrupt_after=['tools'])

    agent = graph()
    assert 'delegation_receipts' not in agent.get_input_jsonschema()['properties']
    await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Build'}]}, config)
    assert (await agent.aget_state(config)).values['delegation_receipts'][0]['tool'] == 'reference'
    # Reconstruct middleware and compiled graph to rule out instance-only caches.
    agent = graph()
    for _ in range(8):
        state = await agent.aget_state(config)
        if not state.next:
            break
        await agent.ainvoke(None, config)
    else:
        pytest.fail('Graph did not finish')
    assert invoked == ['reference', 'review', 'task', 'review', 'task', 'review']
    assert len(received) == 2
    assert 'baseline' in received[0] and 'after-first' in received[1]
    for description in received:
        preserved = description.split('not coordinator prose):\n')[1].split('\nThese bounded')[0]
        assert '"expected":42' in json.loads(preserved)[0]['content_preview']
    final = (await agent.aget_state(config)).values
    assert not any(isinstance(m, ToolMessage) for m in final['messages'])
    # Another conversation cannot inherit any receipts from this middleware.
    fresh = graph()
    assert not (await fresh.aget_state({'configurable': {'thread_id': 'two'}})).values


@pytest.mark.parametrize('parallel', [False, True])
def test_specialist_reads_survive_compaction_without_authorizing_pending_or_failed_reads(parallel):
    from charter.harness.delegation_prerequisites import SpecialistReadPrerequisites
    invoked = []

    @tool
    def specification():
        """Read specification."""
        invoked.append('specification')
        return 'actual source'

    @tool
    def evidence():
        """Read evidence."""
        invoked.append('evidence')
        return 'actual failures'

    @tool
    def write():
        """Write owned source."""
        invoked.append('write')
        return 'saved'

    if parallel:
        first = AIMessage(content='', tool_calls=[
            {'name': 'specification', 'id': 's', 'args': {}},
            {'name': 'evidence', 'id': 'e', 'args': {}},
            {'name': 'write', 'id': 'pending', 'args': {}}])
        responses = [first, call('write', 'ready'), AIMessage(content='done')]
    else:
        responses = [call('specification', 's'), call('write', 'missing-evidence'),
                     call('evidence', 'e'), call('write', 'ready'), AIMessage(content='done')]
    agent = create_agent(ToolModel(responses=responses), tools=[specification, evidence, write],
        middleware=[compacting_middleware(), SpecialistReadPrerequisites(['specification', 'evidence'], ['write'])])
    assert 'specialist_read_receipts' not in agent.get_input_jsonschema()['properties']
    agent.invoke({'messages': [{'role': 'user', 'content': 'Implement'}]})
    assert sorted(invoked) == ['evidence', 'specification', 'write']


def test_command_results_preserve_updates_and_failed_reads_are_not_recorded():
    from langgraph.types import Command
    from charter.harness.delegation_prerequisites import SpecialistReadPrerequisites
    guard = SpecialistReadPrerequisites(['read'], ['write'])
    request = SimpleNamespace(tool_call={'name': 'read', 'id': 'r'}, state={'messages': []})
    for status, text in [('error', 'failure'), ('success', 'Error executing tool read: missing')]:
        original = ToolMessage(content=text, status=status, tool_call_id='r')
        assert guard.wrap_tool_call(request, lambda _: original) is original
    receipt = ToolMessage(content='actual evidence', tool_call_id='r')
    original = Command(update={'messages': [receipt], 'other': 42})
    saved = guard.wrap_tool_call(request, lambda _: original)
    assert saved.update['messages'] == [receipt]
    assert saved.update['other'] == 42
    assert saved.update['specialist_read_receipts'] == [{'tool': 'read', 'tool_call_id': 'r'}]
    assert 'specialist_read_receipts' not in original.update


def test_failed_specialist_completion_requires_new_review_after_compaction():
    guard = DelegationPrerequisites([], ['review'])
    state = {'messages': [], 'delegation_receipts': [
        {'tool': 'review', 'tool_call_id': 'before', 'content_preview': 'passed', 'truncated': False}]}
    request = SimpleNamespace(tool_call={'name': 'task', 'id': 'failed', 'args': {}}, state=state)
    saved = guard.wrap_tool_call(request, lambda _: ToolMessage(
        content='Specialist failed after partial work', status='error', tool_call_id='failed'))
    state['delegation_receipts'].extend(saved.update['delegation_receipts'])
    assert guard._missing(state) == {'review'}


@pytest.mark.parametrize('task_first', [False, True])
def test_review_started_in_same_batch_as_specialist_cannot_verify_its_output(task_first):
    invoked = []

    @tool
    def task():
        """Run specialist."""
        invoked.append('task')
        return 'changed'

    @tool
    def review():
        """Run review."""
        invoked.append('review')
        return 'verified'

    calls = [{'name': 'task', 'id': 'first', 'args': {}},
             {'name': 'review', 'id': 'concurrent', 'args': {}}]
    if not task_first:
        calls.reverse()
    model = ToolModel(responses=[call('review', 'baseline'), AIMessage(content='', tool_calls=calls),
        call('task', 'must-not-run'), call('review', 'fresh'), call('task', 'second'),
        call('review', 'final'), AIMessage(content='done')])
    agent = create_agent(model, tools=[task, review],
        middleware=[compacting_middleware(), DelegationPrerequisites([], ['review'])])
    agent.invoke({'messages': [{'role': 'user', 'content': 'Build'}]})
    assert invoked.count('task') == 2
    assert invoked.count('review') == 4


@pytest.mark.asyncio
async def test_deferred_coordinator_inspection_opens_after_return_not_pending_or_rejected_task():
    from pydantic import Field
    invoked = []

    class Recording(ToolModel):
        offered: list = Field(default_factory=list)

        def bind_tools(self, tools, **kwargs):
            self.offered.append([t.get('name') if isinstance(t, dict) else t.name for t in tools])
            return self

    @tool
    def reference():
        """Read contract."""
        invoked.append('reference')
        return 'contract'

    @tool
    def inspect_source():
        """Inspect implementation."""
        invoked.append('inspect')
        return 'source'

    @tool
    def task():
        """Delegate initial implementation."""
        invoked.append('task')
        return 'changed'

    parallel = AIMessage(content='', tool_calls=[
        {'name': 'task', 'id': 'child', 'args': {}},
        {'name': 'inspect_source', 'id': 'pending', 'args': {}}])
    model = Recording(responses=[call('inspect_source', 'early'), call('task', 'rejected'),
        call('inspect_source', 'still-early'), call('reference', 'ref'), parallel,
        call('inspect_source', 'ready'), AIMessage(content='done')])
    agent = create_agent(model, tools=[reference, inspect_source, task],
        middleware=[compacting_middleware(), DelegationPrerequisites(['reference'], deferred_tools=['inspect_source'])])
    await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Build'}]})
    assert invoked == ['reference', 'task', 'inspect']
    assert all('inspect_source' not in offered for offered in model.offered[:5])
    assert 'inspect_source' in model.offered[5]


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, ident):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': {}, 'id': ident}])


@pytest.mark.asyncio
async def test_coordinator_reminder_tracks_missing_review_and_compaction_without_saved_messages():
    guard = DelegationPrerequisites(['reference'], ['review'])
    history = [call('review', 'r'), ToolMessage(content='Executed: 2 failures', tool_call_id='r')]
    request = SimpleNamespace(state={'messages': history}, messages=history)
    request.override = lambda **changes: SimpleNamespace(**(vars(request) | changes))
    result = guard.wrap_model_call(request, lambda r: r)
    assert len(history) == 2
    assert 'complete: reference.' in result.messages[-1].content
    assert 'complete: reference, review.' not in result.messages[-1].content
    history.extend([call('reference', 'f'), ToolMessage(content='fields', tool_call_id='f')])
    assert guard.wrap_model_call(request, lambda r: r) is request
    history.extend([call('task', 't'), ToolMessage(content='changed', tool_call_id='t')])
    assert 'complete: review.' in guard.wrap_model_call(request, lambda r: r).messages[-1].content
    request.state = {'messages': [AIMessage(content='Summary: all prerequisites passed')]}
    async def identity(value):
        return value
    result = await guard.awrap_model_call(request, identity)
    assert 'complete: reference, review.' in result.messages[-1].content
    assert len(history) == 6


@pytest.mark.asyncio
async def test_coordinator_cannot_finish_after_helper_without_fresh_review():
    guard = DelegationPrerequisites(['reference'], ['review'])
    before = [call('reference', 'ref'), ToolMessage(content='contract', tool_call_id='ref'),
              call('review', 'baseline'), ToolMessage(content='baseline failed', tool_call_id='baseline')]
    guard.after_agent({'messages': before}, None)
    await guard.aafter_agent({'messages': before}, None)
    task = [call('task', 'helper'), ToolMessage(content='source changed', tool_call_id='helper')]
    with pytest.raises(ValueError, match='without fresh review: review'):
        guard.after_agent({'messages': before + task}, None)
    with pytest.raises(ValueError, match='without fresh review: review'):
        await guard.aafter_agent({'messages': before + task}, None)
    reviewed = before + task + [call('review', 'current'),
                               ToolMessage(content='29 pass, 10 fail', tool_call_id='current')]
    guard.after_agent({'messages': reviewed}, None)
    await guard.aafter_agent({'messages': reviewed}, None)
    # A later helper invalidates the earlier review even if chat history compacts.
    retained = {'messages': [AIMessage(content='summary')], 'delegation_receipts': [
        {'tool': 'task', 'tool_call_id': 'helper', 'returned': True},
        {'tool': 'review', 'tool_call_id': 'current', 'after_task_id': 'helper',
         'content_preview': '29 pass, 10 fail', 'truncated': False},
        {'tool': 'task', 'tool_call_id': 'later', 'returned': True}]}
    with pytest.raises(ValueError, match='without fresh review: review'):
        guard.after_agent(retained, None)


@pytest.mark.asyncio
async def test_only_successful_completed_lookup_allows_delegation():
    invoked = []

    @tool
    def task():
        """Delegate work."""
        invoked.append('delegated')
        return 'done'

    @tool
    def reference():
        """Read contract evidence."""
        invoked.append('read')
        return 'contract fields'

    model = ToolModel(responses=[
        call('task', 'early'), call('reference', 'lookup'),
        call('task', 'ready'), AIMessage(content='done')])
    agent = create_agent(model, tools=[task, reference],
                         middleware=[DelegationPrerequisites(['reference'])])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Build'}]})
    assert invoked == ['read', 'delegated']
    outcomes = {m.tool_call_id: m for m in result['messages'] if isinstance(m, ToolMessage)}
    assert outcomes['early'].status == 'error'
    assert outcomes['ready'].status == 'success'


@pytest.mark.parametrize('history', [
    [AIMessage(content='I read reference successfully')],
    [call('reference', 'pending')],
    [call('reference', 'bad'), ToolMessage(content='failed', tool_call_id='bad', status='error')],
    [call('reference', 'bad'), ToolMessage(content='Error executing tool reference: missing file', tool_call_id='bad')],
    [call('reference', 'bad'), ToolMessage(content=[{'type': 'text', 'text': 'Error executing tool reference: missing file'}], tool_call_id='bad')],
    [ToolMessage(content='contract', name='reference', tool_call_id='orphan')],
])
def test_prose_pending_failed_and_uncorrelated_results_do_not_count(history):
    guard = DelegationPrerequisites(['reference'])
    request = SimpleNamespace(tool_call={'name': 'task', 'id': 't'}, state={'messages': history})
    def unexpected(_):
        pytest.fail('delegation must not execute')
    assert guard.wrap_tool_call(request, unexpected).status == 'error'


def test_all_required_results_are_needed_and_other_tools_are_unrestricted():
    guard = DelegationPrerequisites(['first', 'second'])
    history = [call('first', '1'), ToolMessage(content='ok', tool_call_id='1')]
    req = SimpleNamespace(tool_call={'name': 'task', 'id': 't'}, state={'messages': history})
    blocked = guard.wrap_tool_call(req, lambda _: 'unexpected')
    assert 'second' in blocked.content
    history.extend([call('second', '2'), ToolMessage(content='ok', tool_call_id='2')])
    assert guard.wrap_tool_call(req, lambda _: 'executed') == 'executed'
    req.tool_call['name'] = 'reference'
    req.state = {'messages': []}
    assert guard.wrap_tool_call(req, lambda _: 'read') == 'read'


@pytest.mark.asyncio
async def test_parallel_lookup_and_task_cannot_use_a_pending_result():
    invoked = []

    @tool
    def task():
        """Delegate work."""
        invoked.append('delegated')
        return 'done'

    @tool
    def reference():
        """Read contract evidence."""
        invoked.append('read')
        return 'contract fields'

    pending = AIMessage(content='', tool_calls=[
        {'name': 'reference', 'args': {}, 'id': 'read'},
        {'name': 'task', 'args': {}, 'id': 'early'}])
    agent = create_agent(ToolModel(responses=[pending, call('task', 'ready'), AIMessage(content='done')]),
                         tools=[task, reference], middleware=[DelegationPrerequisites(['reference'])])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Build'}]})
    assert invoked == ['read', 'delegated']
    outcomes = {m.tool_call_id: m for m in result['messages'] if isinstance(m, ToolMessage)}
    assert outcomes['early'].status == 'error'
    assert outcomes['ready'].status == 'success'


@pytest.mark.asyncio
async def test_review_runs_before_first_task_and_between_repairs_even_when_assertions_fail():
    invoked = []

    @tool
    def task():
        """Delegate an implementation repair."""
        invoked.append('task')
        return 'files changed'

    @tool
    def review():
        """Execute independent acceptance."""
        invoked.append('review')
        return {'ok': False, 'failed': ['positive request rejected']}

    model = ToolModel(responses=[call('task', 'too-early'), call('review', 'baseline'),
        call('task', 'first'), call('task', 'unreviewed'), call('review', 'current'),
        call('task', 'repair'), call('review', 'final'), AIMessage(content='done')])
    agent = create_agent(model, tools=[task, review], middleware=[DelegationPrerequisites([], ['review'])])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Build'}]})
    assert invoked == ['review', 'task', 'review', 'task', 'review']
    outcomes = {m.tool_call_id: m for m in result['messages'] if isinstance(m, ToolMessage)}
    assert outcomes['too-early'].status == outcomes['unreviewed'].status == 'error'
    assert outcomes['first'].status == outcomes['repair'].status == 'success'


def test_older_or_failed_review_cannot_cover_later_specialist_result():
    guard = DelegationPrerequisites([], ['review'])
    history = [call('review', 'old'), ToolMessage(content='passed', tool_call_id='old'),
               call('task', 'child'), ToolMessage(content='changed', tool_call_id='child'),
               call('review', 'bad'), ToolMessage(content='Error executing tool review: service missing', tool_call_id='bad')]
    request = SimpleNamespace(tool_call={'name': 'task', 'id': 'next'}, state={'messages': history})
    assert guard.wrap_tool_call(request, lambda _: 'must not execute').status == 'error'


@pytest.mark.asyncio
async def test_specialist_receives_actual_review_separately_from_parent_paraphrase():
    received = []

    @tool
    def task(description: str):
        """Delegate a repair."""
        received.append(description)
        return 'done'

    @tool
    def review():
        """Execute a contract check."""
        return {'ok': False, 'field': 'preferences.count', 'expected': 42, 'actual': 1}

    parent = 'Rename the top-level count field.'
    model = ToolModel(responses=[call('review', 'exact-check'),
        AIMessage(content='', tool_calls=[{'name': 'task', 'id': 'child', 'args': {'description': parent}}]),
        call('review', 'final'), AIMessage(content='done')])
    agent = create_agent(model, tools=[task, review], middleware=[DelegationPrerequisites([], ['review'])])
    await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Repair'}]})
    assert len(received) == 1 and received[0].startswith(parent)
    assert 'preferences.count' in received[0] and 'exact-check' in received[0]
    assert 'untrusted diagnostic data' in received[0]
    assert 'not coordinator prose' in received[0]


@pytest.mark.asyncio
async def test_preserved_review_evidence_is_bounded_and_discloses_truncation():
    received = []

    @tool
    def task(description: str):
        """Delegate a repair."""
        received.append(description)
        return 'done'

    @tool
    def review():
        """Return a large diagnostic."""
        return 'exact leading diagnostic ' + '\u03a9' * 10000 + 'OMITTED_TAIL'

    model = ToolModel(responses=[call('review', 'large'),
        AIMessage(content='', tool_calls=[{'name': 'task', 'id': 'child', 'args': {'description': 'Repair'}}]),
        call('review', 'final'), AIMessage(content='done')])
    agent = create_agent(model, tools=[task, review], middleware=[DelegationPrerequisites([], ['review'])])
    await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Repair'}]})
    assert 'exact leading diagnostic' in received[0]
    assert 'OMITTED_TAIL' not in received[0]
    assert '"truncated": true' in received[0]
    assert len(received[0].encode('utf-8')) < 5000


def test_mcp_review_summary_survives_large_earlier_stdout_without_double_wrappers():
    guard = DelegationPrerequisites([], ['review'])
    payload = {'output': 'passing detail ' * 4000, 'ok': False, 'exit_code': 1,
        'report_path': 'saved-report.json',
        'case_summary': {'counts': {'fail': 1}, 'nonpassing_cases': [{'case': 'missing asset', 'actual_count': 0}]}}
    history = [call('review', 'r'), ToolMessage(content=[{'type': 'text', 'text': json.dumps(payload)}], tool_call_id='r')]
    request = SimpleNamespace(state={'messages': history}, tool_call={'name': 'task', 'id': 't', 'args': {'description': 'Repair'}})
    request.override = lambda **changes: SimpleNamespace(**(vars(request) | changes))
    result = guard.wrap_tool_call(request, lambda r: r)
    description = result.tool_call['args']['description']
    records = json.loads(description.split('not coordinator prose):\n', 1)[1].split('\nThese bounded', 1)[0])
    assert len(records) == 1 and records[0]['truncated'] is True
    preview = records[0]['content_preview']
    assert 'saved-report.json' in preview and '"actual_count":0' in preview
    assert '"fail":1' in preview and '"ok":false' in preview
    assert '"type":"text"' not in preview
    assert len(preview.encode('utf-8')) <= 3000
    assert history[1].content[0]['text'] == json.dumps(payload)


from charter.harness.delegation_prerequisites import SpecialistReadPrerequisites


@pytest.mark.asyncio
@pytest.mark.parametrize('parallel', [False, True])
async def test_specialist_must_read_before_writing_in_each_conversation(parallel):
    invoked = []

    @tool
    def write():
        """Edit an owned file."""
        invoked.append('write')
        return 'changed'

    @tool
    def read():
        """Read recorded checks."""
        invoked.append('read')
        return {'ok': False, 'cases': ['response differs']}

    if parallel:
        start = [AIMessage(content='', tool_calls=[
            {'name': 'read', 'args': {}, 'id': 'read'},
            {'name': 'write', 'args': {}, 'id': 'early'}])]
    else:
        start = [call('write', 'early'), call('read', 'read')]
    agent = create_agent(ToolModel(responses=[*start, call('write', 'ready'), AIMessage(content='done')]),
        tools=[write, read], middleware=[SpecialistReadPrerequisites(['read'], ['write'])])
    for _ in range(2):
        result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Repair'}]})
        outcomes = {m.tool_call_id: m for m in result['messages'] if isinstance(m, ToolMessage)}
        assert outcomes['early'].status == 'error'
        assert outcomes['ready'].status == 'success'
    assert invoked == ['read', 'write', 'read', 'write']


@pytest.mark.parametrize('history', [
    [AIMessage(content='Coordinator says read succeeded')],
    [call('read', 'pending')],
    [call('read', 'bad'), ToolMessage(content='failed', tool_call_id='bad', status='error')],
    [call('read', 'bad'), ToolMessage(content='Error executing tool read: missing report', tool_call_id='bad')],
    [call('read', 'bad'), ToolMessage(content=[{'type': 'text', 'text': 'Error executing tool read: missing report'}], tool_call_id='bad')],
    [ToolMessage(content='cases', name='read', tool_call_id='orphan')],
])
def test_specialist_does_not_accept_prose_pending_errors_or_orphan_results(history):
    guard = SpecialistReadPrerequisites(['read'], ['write'])
    request = SimpleNamespace(tool_call={'name': 'write', 'id': 'w'}, state={'messages': history})
    assert guard.wrap_tool_call(request, lambda _: pytest.fail('write must not run')).status == 'error'


def test_specialist_all_reads_required_and_unrelated_tools_remain_available():
    guard = SpecialistReadPrerequisites(['first', 'second'], ['write'])
    history = [call('first', '1'), ToolMessage(content='cases', tool_call_id='1')]
    request = SimpleNamespace(tool_call={'name': 'write', 'id': 'w'}, state={'messages': history})
    assert 'second' in guard.wrap_tool_call(request, lambda _: pytest.fail('must not write')).content
    history.extend([call('second', '2'), ToolMessage(content='cases', tool_call_id='2')])
    assert guard.wrap_tool_call(request, lambda _: 'written') == 'written'
    request.state = {'messages': []}
    request.tool_call['name'] = 'other_read'
    assert guard.wrap_tool_call(request, lambda _: 'read') == 'read'


@pytest.mark.asyncio
async def test_specialist_writers_are_hidden_until_successful_read_without_mutating_tools():
    offered, invoked = [], []

    class RecordingModel(ToolModel):
        def bind_tools(self, tools, **kwargs):
            offered.append({t.get('name') if isinstance(t, dict) else t.name:
                            t.get('description', '') if isinstance(t, dict) else t.description for t in tools})
            return self

    @tool
    def read():
        """Read checks."""
        invoked.append('read')
        return 'Error executing tool read: unavailable' if len(invoked) == 1 else 'Executed checks fail'

    @tool
    def write():
        """Write a file."""
        invoked.append('write')
        return 'saved'

    @tool
    def inspect_source():
        """Read source independently."""
        return 'source'

    original_description = read.description
    model = RecordingModel(responses=[call('read', 'failed'), call('read', 'ready'),
                                     call('write', 'write'), AIMessage(content='done')])
    agent = create_agent(model, tools=[read, write, inspect_source],
                         middleware=[SpecialistReadPrerequisites(['read'], ['write'])])
    await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Repair'}]})
    assert len(offered) == 4
    assert all('write' not in row and row['read'].startswith('Required before editing:') for row in offered[:2])
    assert all('write' in row and row['read'] == original_description for row in offered[2:])
    assert all('inspect_source' in row for row in offered)
    assert read.description == original_description
    assert invoked == ['read', 'read', 'write']


def test_specialist_visibility_handles_dictionary_tools_and_compacted_history():
    guard = SpecialistReadPrerequisites(['read'], ['write'])
    tools = [{'name': 'read', 'description': 'Read checks'}, {'name': 'write', 'description': 'Edit'}]
    history = [call('read', 'r'), ToolMessage(content='cases', tool_call_id='r')]
    request = SimpleNamespace(state={'messages': history}, tools=tools, messages=list(history))
    request.override = lambda **changes: SimpleNamespace(**(vars(request) | changes))
    assert guard.wrap_model_call(request, lambda r: r).tools == tools
    request.state = {'messages': [AIMessage(content='Summary says the read happened')]}
    offered = guard.wrap_model_call(request, lambda r: r).tools
    assert [t['name'] for t in offered] == ['read']
    assert offered[0]['description'].startswith('Required before editing:')
    assert tools[0]['description'] == 'Read checks'
    changed = guard.wrap_model_call(request, lambda r: r)
    assert 'Before editing, complete: read.' in changed.messages[-1].content
    assert 'write tools are hidden' in changed.messages[-1].content
    assert request.messages == history


def test_specialist_reminder_names_only_actual_outstanding_readers():
    guard = SpecialistReadPrerequisites(['read_spec', 'read_cases'], ['write'])
    history = [call('read_spec', 's'), ToolMessage(content='specification', tool_call_id='s')]
    request = SimpleNamespace(state={'messages': history}, tools=[{'name': 'write'},
        {'name': 'read_spec'}, {'name': 'read_cases'}], messages=list(history))
    request.override = lambda **changes: SimpleNamespace(**(vars(request) | changes))
    result = guard.wrap_model_call(request, lambda r: r)
    assert 'complete: read_cases.' in result.messages[-1].content
    assert 'read_spec' not in result.messages[-1].content
    assert request.messages == history
    history.extend([call('read_cases', 'c'), ToolMessage(content='failed cases', tool_call_id='c')])
    assert guard.wrap_model_call(request, lambda r: r) is request
