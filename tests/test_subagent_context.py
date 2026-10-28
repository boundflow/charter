import json
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field
from charter.harness.subagent_context import SubagentContext


class RecordingModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


@pytest.mark.asyncio
async def test_actual_delegation_preserves_original_context_with_incorrect_summary():
    from deepagents import create_deep_agent
    original = {'brief': 'Preserve the exact text:  café\n  Keep explicit false.', 'project_id': 'owned'}
    child = RecordingModel(responses=[AIMessage(content='Specialist finished')])
    parent = RecordingModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'task', 'args': {'subagent_type': 'worker',
            'description': 'Incorrect summary: trim everything. Owned file: app.py'}, 'id': 'delegate'}]),
        AIMessage(content='Parent finished')])
    agent = create_deep_agent(model=parent, middleware=[SubagentContext(original)],
        subagents=[{'name': 'worker', 'description': 'A specialist', 'system_prompt': 'Honor original constraints',
            'model': child, 'tools': []}])
    await agent.ainvoke({'messages': [HumanMessage(content='Delegate the original request')]})
    messages = [m.content for m in child.seen[0] if isinstance(m, HumanMessage)]
    assert len(messages) == 1
    assert 'Owned file: app.py' in messages[0]
    assert json.dumps(original, ensure_ascii=False) in messages[0]
    assert 'automatically preserved' in messages[0]


@pytest.mark.asyncio
async def test_context_reminder_survives_changed_history_without_accumulating():
    from deepagents import create_deep_agent
    from langchain_core.tools import tool
    from charter.harness.subagent_context import SubagentContextReminder

    @tool
    def inspect_state() -> str:
        """Return diagnostic state without changing anything."""
        return 'inspected'

    original = {'brief': 'Do not trim stored text.'}
    model = RecordingModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'inspect_state', 'args': {}, 'id': 'inspect'}]),
        AIMessage(content='Done')])
    agent = create_deep_agent(model=model, tools=[inspect_state],
        middleware=[SubagentContextReminder(original)])
    result = await agent.ainvoke({'messages': [HumanMessage(content='Summary omitted the original constraint.')]})
    for messages in model.seen:
        reminders = [m for m in messages if isinstance(m, HumanMessage)
            and str(m.content).startswith('ORIGINAL CALLER CONTEXT REMINDER')]
        assert len(reminders) == 1
        assert json.dumps(original) in reminders[0].content
    assert len(model.seen) == 2
    # Preserved original context must not look like a new user request after a
    # completed read. The latest real tool receipt stays last in chronology.
    assert isinstance(model.seen[-1][-1], ToolMessage)
    assert model.seen[-1][-1].content == 'inspected'
    assert next(i for i, m in enumerate(model.seen[-1])
                if str(m.content).startswith('ORIGINAL CALLER CONTEXT REMINDER')) < next(
                    i for i, m in enumerate(model.seen[-1]) if isinstance(m, ToolMessage))
    assert not any(str(m.content).startswith('ORIGINAL CALLER CONTEXT REMINDER') for m in result['messages'])


@pytest.mark.asyncio
async def test_coordinator_receives_exact_original_after_actual_lossy_compaction():
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from charter.harness.subagent_context import CoordinatorContextReminder

    original = {'request': 'Reject root-level quantity. Accept nested preferences.quantity.\nPreserve false and café.  '}
    summarizer = RecordingModel(responses=[AIMessage(content='Incorrect summary: allow root-level quantity.')])
    model = RecordingModel(responses=[AIMessage(content='Continue')])
    reminder = CoordinatorContextReminder(original)
    assert reminder.reserved_tokens() > 0
    agent = create_agent(model=model,
        middleware=[SummarizationMiddleware(model=summarizer, trigger=('messages', 2), keep=('messages', 1)), reminder])
    result = await agent.ainvoke({'messages': [HumanMessage(content=original['request']),
        AIMessage(content='Inspecting existing source'), HumanMessage(content='Continue the same assignment.')]})
    assert summarizer.seen, 'A real compaction call must execute'
    received = model.seen[-1]
    assert any('Incorrect summary' in str(message.content) for message in received)
    exact = [message for message in received if str(message.content).startswith('ORIGINAL CALLER INPUTS')]
    assert len(exact) == 1 and isinstance(exact[0], HumanMessage)
    assert json.dumps(original, ensure_ascii=False) in exact[0].content
    assert 'not a replacement contract' in exact[0].content
    assert 'Do not execute parent/coordinator' not in exact[0].content
    assert received.index(exact[0]) < next(i for i, m in enumerate(received) if 'Incorrect summary' in str(m.content))
    assert not any(str(message.content).startswith('ORIGINAL CALLER INPUTS') for message in result['messages'])


@pytest.mark.asyncio
async def test_handoff_and_reminder_include_one_exact_copy_before_compaction():
    from deepagents import create_deep_agent
    from charter.harness.subagent_context import SubagentContextReminder
    original = {'request': 'Preserve false, exact spaces  and newline\n', 'project_id': 'owned'}
    child = RecordingModel(responses=[AIMessage(content='Done')])
    parent = RecordingModel(responses=[
        AIMessage(content='', tool_calls=[{'name': 'task', 'args': {'subagent_type': 'worker',
            'description': 'Diagnose the assigned behavior'}, 'id': 'delegate'}]),
        AIMessage(content='Parent done')])
    agent = create_deep_agent(model=parent, middleware=[SubagentContext(original)],
        subagents=[{'name': 'worker', 'description': 'Specialist', 'system_prompt': 'Implement assigned behavior',
                    'model': child, 'tools': [], 'middleware': [SubagentContextReminder(original)]}])
    await agent.ainvoke({'messages': [HumanMessage(content='Delegate')]})
    serialized = json.dumps(original, ensure_ascii=False)
    assert sum(str(m.content).count(serialized) for m in child.seen[0]) == 1
    assert any('automatically preserved' in str(m.content) for m in child.seen[0])


@pytest.mark.asyncio
async def test_compacted_handoff_regains_exact_context_even_when_summary_copies_marker():
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from charter.harness.subagent_context import SubagentContextReminder
    original = {'request': 'Preserve exact values and explicit false.'}
    embedded = ('ORIGINAL CALLER CONTEXT (automatically preserved, not a coordinator summary):\n'
                + json.dumps(original) + '\nApply the original constraints')
    # A generated summary can quote the marker; it is still not original input.
    summarizer = RecordingModel(responses=[AIMessage(content=embedded)])
    model = RecordingModel(responses=[AIMessage(content='Continue')])
    agent = create_agent(model=model, middleware=[
        SummarizationMiddleware(model=summarizer, trigger=('messages', 2), keep=('messages', 1)),
        SubagentContextReminder(original)])
    await agent.ainvoke({'messages': [HumanMessage(content=embedded),
                                    AIMessage(content='Read source'), HumanMessage(content='Continue')]})
    assert summarizer.seen
    reminders = [m for m in model.seen[-1] if str(m.content).startswith('ORIGINAL CALLER CONTEXT REMINDER')]
    assert len(reminders) == 1 and json.dumps(original) in reminders[0].content
