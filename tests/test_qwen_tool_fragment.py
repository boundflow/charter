import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from charter.harness.qwen_tool_fragment import normalize


SCHEMA = {'type': 'function', 'function': {'name': 'inspect_record', 'parameters': {
    'type': 'object', 'properties': {'text': {'type': 'string'}, 'count': {'type': 'integer', 'minimum': 1},
        'tags': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['text', 'count']}}}
VALID = '<function=inspect_record>\n<parameter=text>\n  exact body\nnext line  \n</parameter>\n<parameter=count>\n2\n</parameter>\n<parameter=tags>\n["one"]\n</parameter>\n</function>\n</tool_call>'


def test_complete_fragment_preserves_text_usage_and_normalization_provenance():
    message = AIMessage(content=VALID, usage_metadata={'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120})
    result = normalize(message, [SCHEMA])
    assert result.tool_calls[0]['args'] == {'text': '  exact body\nnext line  ', 'count': 2, 'tags': ['one']}
    assert result.usage_metadata == message.usage_metadata
    assert result.additional_kwargs['tool_format_normalization']['raw_sha256']
    assert message.content == VALID and not message.tool_calls


@pytest.mark.parametrize('text', [
    'Example:\n' + VALID, '```\n' + VALID + '\n```', VALID + '\nI did it.',
    VALID + VALID, VALID.replace('</tool_call>', ''), VALID.replace('inspect_record', 'unknown'),
    VALID.replace('<parameter=count>\n2', '<parameter=count>\ntrue'),
    VALID.replace('<parameter=count>\n2', '<parameter=count>\n0'),
    VALID.replace('<parameter=tags>', '<parameter=unknown>'),
    VALID.replace('</function>', '<parameter=count>3</parameter></function>'),
    VALID.replace('  exact body', '<function=inspect_record>'),
    VALID.replace('["one"]', '[42]'),
    VALID.replace('<parameter=count>\n2', '<parameter=count>\nNaN'),
])
def test_ambiguous_incomplete_unknown_and_schema_invalid_output_is_not_normalized(text):
    assert normalize(AIMessage(content=text), [SCHEMA]) is None


def test_never_adds_calls_to_already_structured_output_or_tools_not_advertised():
    message = AIMessage(content=VALID, tool_calls=[{'name': 'inspect_record', 'args': {}, 'id': 'native'}])
    assert normalize(message, [SCHEMA]) is None
    assert normalize(AIMessage(content=VALID), []) is None


def test_nested_duplicate_object_keys_are_rejected():
    schema = {'name': 'inspect_record', 'parameters': {'type': 'object', 'properties': {'data': {'type': 'object'}}}}
    text = '<function=inspect_record><parameter=data>{"x":1,"x":2}</parameter></function></tool_call>'
    assert normalize(AIMessage(content=text), [schema]) is None


@pytest.mark.asyncio
async def test_normalized_fragment_runs_through_existing_tool_wrapper_exactly_once():
    from deepagents import create_deep_agent
    from charter.harness.tool_format import ToolFormatRecovery
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
    calls = []

    @tool
    def inspect_record(count: int):
        """Inspect an authorized record."""
        # Represents the ordinary governed tool wrapper; normalization never invokes it directly.
        calls.append(count)
        raise ValueError('Execution denied by the existing tool policy')

    fragment = '<function=inspect_record>\n<parameter=count>1</parameter>\n</function>\n</tool_call>'
    model = ToolModel(responses=[AIMessage(content=fragment), AIMessage(content='Execution was denied')])
    agent = create_deep_agent(model=model, tools=[inspect_record], middleware=[ToolFormatRecovery(0, True)])
    with pytest.raises(ValueError, match='existing tool policy'):
        await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Inspect the record'}]})
    assert calls == [1]
