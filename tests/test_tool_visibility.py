import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field

from charter.harness.tool_visibility import BuiltinToolExclusions


class RecordingModel(FakeMessagesListChatModel):
    offered: list = Field(default_factory=list)
    descriptions: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.offered.append([tool.get('name') if isinstance(tool, dict) else tool.name for tool in tools])
        self.descriptions.append({tool.get('name') if isinstance(tool, dict) else tool.name:
            tool.get('description', '') if isinstance(tool, dict) else tool.description for tool in tools})
        return self


@pytest.mark.asyncio
async def test_actual_deep_agent_hides_scratch_writes_and_rejects_forced_call():
    from deepagents import create_deep_agent
    model = RecordingModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'write_file', 'args': {'file_path': '/incorrect.py', 'content': 'bad'}, 'id': 'forced'}]),
        AIMessage(content='Done'),
    ])
    agent = create_deep_agent(model=model, middleware=[BuiltinToolExclusions(['write_file', 'edit_file'])])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Test routing'}]})
    assert all('write_file' not in tools and 'edit_file' not in tools for tools in model.offered)
    assert any('read_file' in tools for tools in model.offered)  # history/offloaded output remains readable
    receipts = [m for m in result['messages'] if isinstance(m, ToolMessage)]
    assert len(receipts) == 1 and receipts[0].status == 'error'
    assert 'disabled' in receipts[0].content
    assert '/incorrect.py' not in result.get('files', {})


@pytest.mark.asyncio
async def test_inline_specialist_receives_the_same_exclusions():
    from deepagents import create_deep_agent
    child = RecordingModel(responses=[AIMessage(content='Specialist finished')])
    parent = RecordingModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'task', 'args': {'subagent_type': 'worker', 'description': 'Inspect available tools'}, 'id': 'delegate'}]),
        AIMessage(content='Parent finished'),
    ])
    agent = create_deep_agent(model=parent,
        middleware=[BuiltinToolExclusions(['write_file', 'edit_file'])],
        subagents=[{'name': 'worker', 'description': 'A specialist', 'system_prompt': 'Inspect interfaces',
                    'model': child, 'tools': [], 'middleware': [BuiltinToolExclusions(['write_file', 'edit_file'])]}])
    await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Delegate'}]})
    assert child.offered
    assert all('write_file' not in tools and 'edit_file' not in tools for tools in child.offered)


@pytest.mark.asyncio
async def test_scratch_read_failure_explains_the_correct_project_reader():
    from deepagents import create_deep_agent
    hint = 'Scratch history only. Persistent files require the declared project reader; do not guess alternate scratch paths.'
    model = RecordingModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'read_file', 'args': {'file_path': '/app.py'}, 'id': 'wrong'}]),
        AIMessage(content='Use project reader')])
    agent = create_deep_agent(model=model, middleware=[BuiltinToolExclusions([], {'read_file': hint})])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Inspect file'}]})
    assert model.descriptions[0]['read_file'].startswith(hint)
    receipt = next(m for m in result['messages'] if isinstance(m, ToolMessage))
    assert 'not found' in receipt.content and hint in receipt.content


@pytest.mark.asyncio
async def test_offloaded_tool_result_remains_readable_while_scratch_writes_stay_disabled():
    from deepagents import create_deep_agent
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain_core.tools import tool

    @tool
    def source():
        """Read a large source result."""
        return '\n'.join(f'original evidence line {n}' for n in range(1000))

    model = RecordingModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'source', 'args': {}, 'id': 'large'}]),
        AIMessage(content='', tool_calls=[{'name': 'read_file',
            'args': {'file_path': '/large_tool_results/large', 'offset': 450, 'limit': 2}, 'id': 'recover'}]),
        AIMessage(content='', tool_calls=[{'name': 'write_file',
            'args': {'file_path': '/product.py', 'content': 'bypass'}, 'id': 'blocked'}]),
        AIMessage(content='done')])
    agent = create_deep_agent(model=model, tools=[source], middleware=[
        FilesystemMiddleware(tool_token_limit_before_evict=1000),
        BuiltinToolExclusions(['write_file', 'edit_file'], {
            'read_file': 'Read harness-offloaded results/history using the exact returned path and bounded integer offset/limit. Persistent project files require the project reader.'})])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Inspect evidence'}]})
    receipts = {m.tool_call_id: m for m in result['messages'] if isinstance(m, ToolMessage)}
    assert 'saved in the filesystem' in receipts['large'].content
    assert 'original evidence line 450' in receipts['recover'].content
    assert 'original evidence line 451' in receipts['recover'].content
    assert 'original evidence line 452' not in receipts['recover'].content
    assert receipts['blocked'].status == 'error'
    assert '/product.py' not in result.get('files', {})
    assert all('read_file' in offered and 'write_file' not in offered for offered in model.offered)
