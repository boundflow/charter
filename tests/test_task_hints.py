from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field
import pytest

from charter.config.agent import AgentConfig
from charter.harness.tool_visibility import BuiltinToolExclusions


class RecordingModel(FakeMessagesListChatModel):
    descriptions: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.descriptions.append({
            t.get('name') if isinstance(t, dict) else t.name:
            t.get('description', '') if isinstance(t, dict) else t.description
            for t in tools})
        return self


@pytest.mark.asyncio
async def test_configured_task_hint_reaches_model_and_preserves_delegation():
    from deepagents import create_deep_agent
    hint = 'Choose the specialist whose declared capability matches this assignment.'
    cfg = AgentConfig.model_validate({
        'apiVersion': 'charter/v1', 'kind': 'AgentConfig', 'name': 'routing',
        'version': 1, 'model': 'test-model', 'objective': 'Delegate',
        'builtin_tool_hints': {'task': hint}})
    parent = RecordingModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'task', 'args': {
            'subagent_type': 'worker', 'description': 'Inspect assigned evidence'}, 'id': 'delegate'}]),
        AIMessage(content='Finished')])
    child = RecordingModel(responses=[AIMessage(content='Evidence inspected')])
    agent = create_deep_agent(model=parent,
        middleware=[BuiltinToolExclusions([], cfg.builtin_tool_hints)],
        subagents=[{'name': 'worker', 'description': 'Inspect evidence',
                    'system_prompt': 'Read only', 'model': child, 'tools': []}])
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Delegate inspection'}]})
    assert parent.descriptions[0]['task'].startswith(hint)
    assert 'worker' in parent.descriptions[0]['task']
    receipt = next(m for m in result['messages'] if isinstance(m, ToolMessage)
                   and m.tool_call_id == 'delegate')
    assert receipt.status == 'success'
    assert 'Evidence inspected' in str(receipt.content)
