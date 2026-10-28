import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from charter.harness.tool_format import MalformedToolOutput, ToolFormatRecovery


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_printed_tool_is_not_executed_and_structured_retry_runs_once():
    from deepagents import create_deep_agent
    calls = []

    @tool
    def inspect_record():
        """Inspect a record."""
        calls.append('actual')
        return 'record inspected'

    model = ToolModel(responses=[
        AIMessage(content='<function=inspect_record>\n</function>\n</tool_call>'),
        AIMessage(content='', tool_calls=[{'name': 'inspect_record', 'args': {}, 'id': 'real'}]),
        AIMessage(content='Inspected the record'),
    ])
    agent = create_deep_agent(model=model, tools=[inspect_record], middleware=[ToolFormatRecovery(1)])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Inspect it'}]})
    assert calls == ['actual']
    assert sum(isinstance(m, ToolMessage) for m in result['messages']) == 1
    assert result['messages'][-1].content == 'Inspected the record'


@pytest.mark.asyncio
async def test_repeated_markup_fails_without_running_any_tool():
    from deepagents import create_deep_agent
    calls = []

    @tool
    def inspect_record():
        """Inspect a record."""
        calls.append('unexpected')
        return 'record'

    model = ToolModel(responses=[AIMessage(content='<function=inspect_record></function>')])
    agent = create_deep_agent(model=model, tools=[inspect_record], middleware=[ToolFormatRecovery(1)])
    with pytest.raises(MalformedToolOutput, match='no printed action was executed'):
        await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Inspect it'}]})
    assert not calls


@pytest.mark.asyncio
async def test_ordinary_text_and_unknown_function_examples_do_not_trigger_retry():
    from deepagents import create_deep_agent
    text = 'Example documentation: <function=unknown_example></function>'
    model = ToolModel(responses=[AIMessage(content=text)])
    agent = create_deep_agent(model=model, middleware=[ToolFormatRecovery(1)])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Explain the notation'}]})
    assert result['messages'][-1].content == text
