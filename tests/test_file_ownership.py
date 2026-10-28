import asyncio
from dataclasses import dataclass, replace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.middleware.types import ModelRequest
from langchain_core.tools import tool
from pydantic import Field

from charter.harness.file_ownership import FileOwnership, OwnedWrites


class ToolModel(FakeMessagesListChatModel):
    observed_tools: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.observed_tools[:] = tools
        return self


@dataclass
class Request:
    tool_call: dict

    def override(self, **kwargs):
        return replace(self, **kwargs)


def task_request(files, ident='task'):
    return Request({'name': 'task', 'id': ident, 'args': {
        'description': 'Do the assigned work', 'subagent_type': 'writer', 'owned_files': files}})


@pytest.mark.asyncio
async def test_real_inline_specialist_can_write_only_its_declared_files():
    from deepagents import create_deep_agent
    writes = []

    @tool
    def save(path: str, content: str):
        """Save a project file."""
        writes.append((path, content))
        return 'saved'

    child = ToolModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'save', 'id': 'outside',
            'args': {'path': 'other.py', 'content': 'must not write'}}]),
        AIMessage(content='', tool_calls=[{'name': 'save', 'id': 'inside',
            'args': {'path': 'owned.py', 'content': 'assigned'}}]),
        AIMessage(content='Assigned file saved; other write refused')])
    parent = ToolModel(responses=[AIMessage(content='', tool_calls=[task_request(['owned.py']).tool_call]),
                                 AIMessage(content='done')])
    owner = FileOwnership({'save': ['path']}, 4, ['writer'])
    agent = create_deep_agent(model=parent, middleware=[owner], subagents=[{
        'name': 'writer', 'description': 'Write assigned files', 'system_prompt': 'Do assigned work',
        'model': child, 'tools': [save], 'middleware': [OwnedWrites(owner)]}])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Build'}]})
    assert writes == [('owned.py', 'assigned')]
    assert result['messages'][-1].content == 'done'
    task = next(t for t in parent.observed_tools if t.name == 'task')
    assert 'owned_files' in task.args_schema['required']
    assert task.args_schema['properties']['subagent_type']['enum'] == ['writer']
    assert task.args_schema['properties']['subagent_type'].get('default') != 'general-purpose'
    assert owner.scope.get() is None and owner.active == set()


@pytest.mark.asyncio
async def test_concurrent_scopes_reject_overlap_and_release_after_failure():
    owner = FileOwnership({'save': ['path', 'file_path']}, 4, ['writer'])
    guard = OwnedWrites(owner)
    started, release = asyncio.Event(), asyncio.Event()

    async def first(_request):
        assert owner.scope.get() == frozenset({'a.py'})
        started.set()
        await release.wait()
        raise RuntimeError('worker failed')

    running = asyncio.create_task(owner.awrap_tool_call(task_request(['a.py']), first))
    await started.wait()

    async def unexpected(_request):
        pytest.fail('overlapping task must not run')

    denied = await owner.awrap_tool_call(task_request(['a.py'], 'overlap'), unexpected)
    assert denied.status == 'error' and 'already owned' in denied.content

    async def second(_request):
        assert owner.scope.get() == frozenset({'b.py'})
        req = Request({'name': 'save', 'id': 'write', 'args': {'file_path': 'b.py'}})
        assert guard.wrap_tool_call(req, lambda _: 'saved') == 'saved'
        req.tool_call['args'] = {'path': 'a.py'}
        assert guard.wrap_tool_call(req, lambda _: 'unexpected').status == 'error'
        return 'second finished'

    assert await owner.awrap_tool_call(task_request(['b.py'], 'second'), second) == 'second finished'
    release.set()
    with pytest.raises(RuntimeError, match='worker failed'):
        await running
    assert owner.active == set() and owner.scope.get() is None
    assert owner.wrap_tool_call(task_request(['a.py']), lambda _: 'retry') == 'retry'


@pytest.mark.parametrize('paths', [None, 'a.py', ['../a.py'], ['/a.py'], ['a.py', 'a.py'],
                                  ['a', 'b', 'c', 'd', 'e'], ['C:/a.py'], [''], [1]])
def test_invalid_assignments_never_delegate(paths):
    owner = FileOwnership({'save': ['path']}, 4, ['writer'])
    result = owner.wrap_tool_call(task_request(paths), lambda _: pytest.fail('must refuse'))
    assert result.status == 'error'
    assert owner.active == set()


def test_readonly_missing_scope_and_conflicting_aliases_cannot_write():
    owner = FileOwnership({'save': ['path', 'file_path']}, 4, ['writer'])
    guard = OwnedWrites(owner)
    request = Request({'name': 'save', 'id': 'w', 'args': {'path': 'a.py'}})
    assert guard.wrap_tool_call(request, lambda _: 'bad').status == 'error'
    assert owner.wrap_tool_call(task_request([]),
        lambda _: guard.wrap_tool_call(request, lambda _: 'bad')).status == 'error'
    request.tool_call['args']['file_path'] = 'b.py'
    assert owner.wrap_tool_call(task_request(['a.py', 'b.py']),
        lambda _: guard.wrap_tool_call(request, lambda _: 'bad')).status == 'error'


def test_misspelled_destination_is_not_reported_as_an_ownership_denial():
    owner = FileOwnership({'copy_asset': ['destination']}, 4, ['writer'])
    guard = OwnedWrites(owner)
    request = Request({'name': 'copy_asset', 'id': 'w',
                       'args': {'destination_path': 'new/font.woff2'}})
    def delegated(_):
        denied = guard.wrap_tool_call(request, lambda _: pytest.fail('No path supplied'))
        assert denied.status == 'error'
        assert 'no configured destination argument' in denied.content
        assert 'expected a nonempty path in one of: destination' in denied.content
        assert 'outside' not in denied.content
        assert owner.scope.get() == frozenset({'new/font.woff2'})
        request.tool_call['args'] = {'destination': 'new/font.woff2'}
        assert guard.wrap_tool_call(request, lambda _: 'saved') == 'saved'
        request.tool_call['args'] = {'destination': 'other/font.woff2'}
        denied = guard.wrap_tool_call(request, lambda _: pytest.fail('Still outside ownership'))
        assert denied.status == 'error' and 'outside' in denied.content
    owner.wrap_tool_call(task_request(['new/font.woff2']), delegated)


def test_current_scope_survives_lost_task_text_without_mutating_messages_or_policy():
    owner = FileOwnership({'save': ['path']}, 4, ['writer'])
    guard = OwnedWrites(owner)
    request = ModelRequest(model=ToolModel(responses=[]), messages=[AIMessage(content='Historical plan: other.py')],
                           system_prompt='Standing permissions', tools=[])
    assert guard._model_request(request) is request
    def work(_):
        view = guard._model_request(request)
        assert view.system_message.content.startswith('Standing permissions')
        assert 'CURRENT SPECIALIST WRITE SCOPE' in view.system_message.content
        assert '["owned.py"]' in view.system_message.content
        assert 'historical plan does not change' in view.system_message.content
        assert 'claims no successful edit or check' in view.system_message.content
        assert view.messages == request.messages
        assert request.system_message.content == 'Standing permissions'
        denied = guard.wrap_tool_call(Request({'name':'save','id':'outside','args':{'path':'other.py'}}),
                                      lambda _: pytest.fail('outside write executed'))
        assert '["owned.py"]' in denied.content
        assert guard.wrap_tool_call(Request({'name':'save','id':'inside','args':{'path':'owned.py'}}),
                                    lambda _: 'saved') == 'saved'
    owner.wrap_tool_call(task_request(['owned.py']), work)
    assert guard._model_request(request) is request


def test_displayed_paths_preserve_assignment_case_and_release_with_actual_claim():
    owner = FileOwnership({'save':['path']},4,['writer'])
    guard = OwnedWrites(owner)
    request = ModelRequest(model=ToolModel(responses=[]),messages=[],system_prompt='Standing permissions',tools=[])
    def work(_):
        assert owner.display_scope.get() == ('ui/ConfigurationPanel.tsx',)
        text = guard._model_request(request).system_message.content
        assert 'ui/ConfigurationPanel.tsx' in text and 'ui/configurationpanel.tsx' not in text
        denied = guard.wrap_tool_call(Request({'name':'save','id':'bad','args':{'path':'other.py'}}),lambda _:pytest.fail('Unowned'))
        assert 'ui/ConfigurationPanel.tsx' in denied.content
        raise RuntimeError('test release')
    with pytest.raises(RuntimeError,match='test release'):
        owner.wrap_tool_call(task_request(['ui/ConfigurationPanel.tsx']),work)
    assert owner.scope.get() is None and owner.display_scope.get() is None and not owner.active


@pytest.mark.asyncio
async def test_cancellation_releases_ownership():
    owner = FileOwnership({'save': ['path']}, 4, ['writer'])
    started = asyncio.Event()

    async def work(_request):
        started.set()
        await asyncio.Future()

    running = asyncio.create_task(owner.awrap_tool_call(task_request(['a.py']), work))
    await started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert owner.active == set() and owner.scope.get() is None


@pytest.mark.asyncio
async def test_interrupted_inline_task_restores_scope_on_checkpoint_resume():
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command, interrupt
    writes = []

    @tool
    def save(path: str):
        """Save a file after an external decision."""
        answer = interrupt('Proceed with the test write?')
        if answer == 'yes':
            writes.append(path)
        return answer

    child = ToolModel(responses=[AIMessage(content='', tool_calls=[
        {'name': 'save', 'id': 'save', 'args': {'path': 'owned.py'}}]), AIMessage(content='done')])
    parent = ToolModel(responses=[AIMessage(content='', tool_calls=[task_request(['owned.py']).tool_call]),
                                 AIMessage(content='done')])
    owner = FileOwnership({'save': ['path']}, 4, ['writer'])
    agent = create_deep_agent(model=parent, middleware=[owner], checkpointer=InMemorySaver(), subagents=[{
        'name': 'writer', 'description': 'Write assigned file', 'system_prompt': 'Do assigned work',
        'model': child, 'tools': [save], 'middleware': [OwnedWrites(owner)]}])
    config = {'configurable': {'thread_id': 'ownership-resume-test'}}
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Build'}]}, config)
    assert result.get('__interrupt__') and writes == []
    assert owner.active == set() and owner.scope.get() is None
    result = await agent.ainvoke(Command(resume='yes'), config)
    assert not result.get('__interrupt__')
    assert writes == ['owned.py']
    assert owner.active == set() and owner.scope.get() is None
