import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field, ValidationError
from charter.harness.assignment_context import AssignmentContext, _outcomes
from charter.harness.delegation_prerequisites import DelegationPrerequisites


def receipt(ok=False, **updates):
    return {'profile': 'api-contract', 'ok': ok, 'exit_code': 0 if ok else 1,
            'report_path': '/evidence/check.json', 'case_summary': {'counts': {'pass' if ok else 'fail': 1}},
            **updates}


def call(name, ident, **args):
    return AIMessage(content='', tool_calls=[{'name': name, 'id': ident, 'args': args}])


class Model(FakeMessagesListChatModel):
    tool_views: list = Field(default_factory=list)
    message_views: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.tool_views.append([t.get('name') if isinstance(t, dict) else t.name for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.message_views.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.parametrize('value', [
    'assertion failed', {'ok': False}, receipt(stale_result='source changed'),
    receipt(timed_out=True), receipt(exit_code=2), receipt(exit_code=True),
    receipt(service_source_matches_current=False), receipt(harness_stable_during_check=False),
    receipt(case_summary='bad shape'), receipt(case_summary={'counts': {'fail': True}}),
    receipt(source_evidence={'stable_during_check': False}), receipt(report_path=12),
    receipt(True, exit_code=1), receipt(False, exit_code=0),
    {'all_profiles_executed': False, 'profiles': [receipt()]},
    {'execution_error': 'unavailable', 'profiles': [receipt()]},
    {'shutdown_error': 'unavailable', 'profiles': [receipt()]},
])
def test_incomplete_stale_or_infrastructure_evidence_never_activates(value):
    assert _outcomes(json.dumps(value)) == []


def test_real_wrapped_receipts_and_latest_profile_outcome():
    wrapped = [{'type': 'text', 'text': json.dumps({'all_profiles_executed': True, 'profiles': [receipt()]})}]
    assert _outcomes(wrapped) == [{'profile': 'api-contract', 'failed': True}]
    guard = AssignmentContext('original request', ['reference'], ['review'])
    state = {'messages': [call('reference', 'r'), ToolMessage(content='contract', tool_call_id='r'),
                          call('review', 'bad'), ToolMessage(content=json.dumps(receipt()), tool_call_id='bad')]}
    assert guard._ready(state)
    state['messages'] += [call('review', 'good'), ToolMessage(content=json.dumps(receipt(True)), tool_call_id='good')]
    assert not guard._ready(state)
    state['messages'] += [call('review', 'other'), ToolMessage(content=json.dumps(receipt(profile='other')), tool_call_id='other')]
    assert guard._ready(state)


def test_assignment_view_preserves_caller_decisions_and_durable_history_after_compaction():
    guard = AssignmentContext('Original exact request: preserve content', ['reference'], ['review'], ['listing'])
    state = {'messages': [], 'assignment_receipts': [
        {'tool': 'reference', 'tool_call_id': 'r'},
        {'tool': 'listing', 'tool_call_id': 'l', 'text': '["api.py"]', 'truncated': False},
        {'tool': 'review', 'tool_call_id': 'v', 'text': 'actual failed check',
         'outcomes': [{'profile': 'api', 'failed': True}]}]}
    messages = [HumanMessage(content='Generated summary: inspect everything again', additional_kwargs={'lc_source': 'summarization'}),
                HumanMessage(content='User clarification: preserve custom records'),
                call('read', 'bad'), ToolMessage(content='Error executing tool read: task required', tool_call_id='bad')]
    request = ModelRequest(model=Model(responses=[]), messages=messages, state=state,
        system_prompt='Standing permissions', tools=[{'name': 'task'}, {'name': 'read'}])
    view = guard._model_request(request)
    assert view.system_message.content.startswith('Standing permissions')
    assert 'CURRENT CONFIGURED PHASE: ASSIGN IMPLEMENTATION' in view.system_message.content
    assert request.system_message.content == 'Standing permissions'
    assert view.tools == [{'name': 'task'}]
    text = '\n'.join(str(m.content) for m in view.messages)
    assert 'Original exact request: preserve content' in text
    assert 'User clarification: preserve custom records' in text
    assert 'api.py' in text and 'task required' in text
    assert 'Generated summary: inspect everything again' not in text
    assert request.messages == messages and state['messages'] == []
    state['assignment_receipts'].append({'tool': 'task', 'tool_call_id': 'handoff'})
    assert guard._model_request(request) is request


def test_prose_cannot_finish_an_unassigned_failed_component():
    guard = AssignmentContext('original', [], ['review'])
    state = {'messages': [call('review', 'v'), ToolMessage(content=json.dumps(receipt()), tool_call_id='v')]}
    with pytest.raises(ValueError, match='before the configured assignment'):
        guard.after_agent(state, None)


@pytest.mark.asyncio
async def test_requested_implementation_assigns_after_reads_without_inventing_failure_and_still_reviews():
    from deepagents import create_deep_agent
    invoked = []

    @tool
    def reference():
        """Read immutable contract."""
        invoked.append('reference')
        return 'requested feature and preservation contract'

    @tool
    def listing():
        """List actual source paths."""
        invoked.append('listing')
        return ['source.py']

    @tool
    def review():
        """Execute independent verification."""
        invoked.append('review')
        return receipt(True)

    parent = Model(responses=[call('listing','files'), call('reference','spec'),
        call('task','handoff',subagent_type='writer',description='Implement original component and preserve content'),
        AIMessage(content='Premature answer'), call('review','current'), AIMessage(content='Current check passed')])
    child = Model(responses=[AIMessage(content='Requested implementation inspected; returned actual work')])
    contract = 'Complete the requested UI component.\nPreserve existing API contracts and café content.'
    stage = AssignmentContext('Original user implementation request', ['reference','listing'], ['review'],
                              ['listing'], requires_failure=False, contract=contract)
    agent = create_deep_agent(model=parent, tools=[reference,listing,review],
        middleware=[DelegationPrerequisites(['reference','listing'],['review'],
                    final_review_retries=2,initial_review_required=False),stage],
        subagents=[{'name':'writer','description':'Implement','system_prompt':'Do assigned implementation',
                    'model':child,'tools':[]}])
    result = await agent.ainvoke({'messages':[{'role':'user','content':'Original user implementation request'}]})
    assert invoked == ['listing','reference','review']
    assert parent.tool_views[2] == ['task']
    view = str(parent.message_views[2])
    assert 'No passing or failing check is inferred' in view
    assert 'a real independent check recorded failing assertions' not in view.lower()
    assert result['messages'][-1].content == 'Current check passed'
    assert contract in '\n'.join(m.content for m in child.message_views[0] if isinstance(m.content,str))
    assert 'Implement original component and preserve content' not in str(child.message_views[0])


def test_request_stage_cannot_bypass_required_reads():
    stage = AssignmentContext('Implement', ['reference','listing'], ['review'], requires_failure=False)
    state = {'messages':[call('reference','r'),ToolMessage(content='contract',tool_call_id='r')]}
    assert not stage._ready(state)
    state['messages'] += [call('listing','l'),ToolMessage(content='source.py',tool_call_id='l')]
    assert stage._ready(state)


def test_bound_child_contract_excludes_parent_plan_wrapper_but_retains_human_clarification():
    initial = 'Coordinator steps: retrieve historical plan with invented old-name.py'
    contract = 'Implement the requested component; preserve original interfaces.'
    stage = AssignmentContext(initial, [], [], requires_failure=False, contract=contract)
    state = {'messages': [HumanMessage(content=initial),
        HumanMessage(content='Keep custom records and café text.'),
        HumanMessage(content='Generated summary: write old-name.py', additional_kwargs={'lc_source':'summarization'})]}
    from types import SimpleNamespace
    from dataclasses import dataclass, replace
    @dataclass
    class Request:
        state: dict
        tool_call: dict
        def override(self, **updates):
            return replace(self, **updates)
    request = Request(state, {'name':'task','id':'child','args':{'description':'Generated design: use wrong endpoint'}})
    child = stage._tool_request(request)
    text = child.tool_call['args']['description']
    assert contract in text and 'Keep custom records and café text.' in text
    assert initial not in text and 'Generated summary' not in text and 'wrong endpoint' not in text
    assert request.state == state and request.tool_call['args']['description'].startswith('Generated design')


@pytest.mark.asyncio
async def test_actual_checkpoint_resume_and_compaction_keep_assignment_evidence_private():
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES
    from langgraph.checkpoint.memory import InMemorySaver
    invoked = []

    class Compact(AgentMiddleware):
        def before_model(self, state, runtime):
            if any(isinstance(m, ToolMessage) for m in state['messages']):
                return {'messages': [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(
                    content='Lossy summary: inspect everything again', additional_kwargs={'lc_source':'summarization'})]}

        async def abefore_model(self, state, runtime):
            return self.before_model(state, runtime)

    @tool
    def reference():
        """Read actual requirements."""
        invoked.append('reference')
        return 'contract'

    @tool
    def review():
        """Execute independent acceptance."""
        invoked.append('review')
        return receipt()

    @tool
    def task(description: str):
        """Run the assigned specialist."""
        invoked.append('task')
        return 'changed source'

    model = Model(responses=[call('reference','ref'), call('review','baseline'),
        call('task','handoff',description='Full original component'), call('review','current'), AIMessage(content='Unresolved')])
    agent = create_agent(model, tools=[reference,review,task], checkpointer=InMemorySaver(),
        middleware=[Compact(), DelegationPrerequisites(['reference'],['review']),
                    AssignmentContext('Exact original component and exclusions', ['reference'], ['review'])],
        interrupt_after=['tools'])
    config = {'configurable':{'thread_id':'resumed-assignment'}}
    result = await agent.ainvoke({'messages':[{'role':'user','content':'Exact original component and exclusions'}]},config)
    for _ in range(5):
        if not (await agent.aget_state(config)).next:
            break
        result = await agent.ainvoke(None,config)
    assert invoked == ['reference','review','task','review']
    assert model.tool_views[2] == ['task']
    assert 'Exact original component and exclusions' in str(model.message_views[2])
    assert 'Lossy summary: inspect everything again' not in str(model.message_views[2])
    assert 'assignment_receipts' not in result
    retained = (await agent.aget_state(config)).values
    assert {r['tool_call_id'] for r in retained['assignment_receipts']} == {'ref','baseline','handoff','current'}


@pytest.mark.asyncio
async def test_real_deepagent_stage_ownership_child_reads_and_fresh_parent_review():
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from charter.harness.file_ownership import FileOwnership, OwnedWrites
    invoked, writes = [], []

    @tool
    def reference():
        """Read actual requirements."""
        invoked.append('reference')
        return 'complete bounded contract'

    @tool
    def listing():
        """List project paths."""
        return ['owned.py', 'other.py']

    @tool
    def review():
        """Execute real independent check."""
        invoked.append('review')
        return receipt()

    @tool
    def inspect():
        """Read source; coordinator should assign instead after failure."""
        invoked.append('inspect')
        return 'source'

    @tool
    def save(path: str, content: str):
        """Write only assigned paths."""
        writes.append((path, content))
        return 'saved'

    parent = Model(responses=[call('listing', 'files'), call('reference', 'ref'), call('review', 'before'),
        call('inspect', 'must-not-run'), call('task', 'handoff', subagent_type='writer',
            description='Complete component; preserve content', owned_files=['owned.py']),
        AIMessage(content='Premature answer'), call('review', 'after'), AIMessage(content='Still failing; report exact scope')])
    child = Model(responses=[call('inspect', 'child-read'), call('save', 'outside', path='other.py', content='forbidden'),
                            call('save', 'owned', path='owned.py', content='assigned'), AIMessage(content='assigned source changed')])
    owner = FileOwnership({'save': ['path']}, 4, ['writer'])
    stage = AssignmentContext('Exact original user contract', ['reference'], ['review'], ['listing'],
                              routing_retries=2, contract='Full original component contract')
    agent = create_deep_agent(model=parent, tools=[reference, listing, review, inspect],
        middleware=[owner, DelegationPrerequisites(['reference'], ['review'], final_review_retries=2), stage],
        subagents=[{'name': 'writer', 'description': 'Implement', 'system_prompt': 'Read and implement owned source',
                    'model': child, 'tools': [inspect, save], 'middleware': [OwnedWrites(owner)]}],
        checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'assignment'}}
    result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Exact original user contract'}]}, config)
    assert invoked == ['reference', 'review', 'inspect', 'review']  # Only the child inspected.
    assert writes == [('owned.py', 'assigned')]
    assert parent.tool_views[3] == ['task'] and parent.tool_views[4] == ['task']
    assert 'review' in parent.tool_views[-1]
    assert result['messages'][-1].content == 'Still failing; report exact scope'
    retained = (await agent.aget_state(config)).values
    assert retained['assignment_receipts'] and 'assignment_receipts' not in result
    assert 'assignment_receipts' not in agent.get_input_jsonschema()['properties']
    assert any(r['tool'] == 'review' and r['after_task_id'] == 'handoff' for r in retained['delegation_receipts'])
    assert owner.scope.get() is None and owner.active == set()
    assert 'assignment_route_failures' not in result
    assert 'assignment_route_failures' not in agent.get_input_jsonschema()['properties']
    assert len(set(retained['assignment_route_failures'])) == 1
    assert 'Full original component contract' in str(child.message_views[0])
    assert 'api-contract' in str(child.message_views[0])  # Actual review evidence retained.


@pytest.mark.asyncio
async def test_unoffered_calls_are_bounded_per_model_turn_including_parallel_calls():
    invoked = []

    @tool
    def reference():
        """Read actual contract."""
        return 'contract'

    @tool
    def review():
        """Execute actual check."""
        return receipt()

    @tool
    def task(description: str):
        """Run actual specialist."""
        invoked.append('task')
        return 'changed'

    parallel = AIMessage(content='',tool_calls=[{'name':'unoffered_a','id':'a','args':{}},
                                                {'name':'unoffered_b','id':'b','args':{}}])
    model = Model(responses=[call('reference','r'),call('review','v'),parallel,
                            call('task','t',description='component'),call('review','after'),AIMessage(content='done')])
    stage = AssignmentContext('original',['reference'],['review'],routing_retries=1)
    agent = create_agent(model,tools=[reference,review,task],middleware=[stage])
    result = await agent.ainvoke({'messages':[{'role':'user','content':'original'}]})
    assert invoked == ['task'] and result['messages'][-1].content == 'done'
    # A second rejected model turn exhausts the same configured one retry.
    model = Model(responses=[call('reference','r'),call('review','v'),parallel,
                            call('unoffered_c','c'),call('task','never',description='must not execute')])
    agent = create_agent(model,tools=[reference,review,task],middleware=[stage])
    with pytest.raises(ValueError,match='after 2 rejected model turns'):
        await agent.ainvoke({'messages':[{'role':'user','content':'original'}]})
    assert invoked == ['task']


def test_default_false_wiring_and_configuration_validation():
    from charter.config.agent import AgentConfig
    from charter.config.loader import load_agent
    from charter.workflows.loop import Loop
    bundle = load_agent(Path(__file__).parents[1] / 'examples/refund-triage')
    raw = bundle.latest.model_dump(by_alias=True)
    for bad in ['absent', 'customer_id']:
        with pytest.raises(ValidationError, match='declared boolean input'):
            AgentConfig.model_validate({**raw, 'coordinator_assignment_input': bad})
    with pytest.raises(ValidationError, match='requires coordinator_assignment_input'):
        AgentConfig.model_validate({**raw, 'coordinator_assignment_context_tools': ['support__create_refund']})
    raw.update(inputs={**raw['inputs'], 'repair': {'type': 'boolean', 'default': False}},
        coordinator_assignment_input='repair', coordinator_assignment_context_tools=['support__create_refund'],
        subagent_review_tools=['support__create_refund'],
        subagents=[{'name': 'writer', 'description': 'Repair', 'prompt': 'Inspect'}])
    cfg = AgentConfig.model_validate(raw)
    with pytest.raises(ValidationError, match='must name declared tools'):
        AgentConfig.model_validate({**raw, 'coordinator_assignment_context_tools':['support__missing']})
    loop = Loop(cfg, bundle.runtime, tools=SimpleNamespace(langchain_tools=lambda: []),
                chat_model=lambda _: object(), store_url='postgresql://unused')
    harness = SimpleNamespace(wiring={'subagents': [{'name': 'writer'}]})
    inputs = {k: s.default if s.default is not None else 'value' for k, s in cfg.inputs.items()}
    assert not any(isinstance(m, AssignmentContext) for m in loop._wiring(harness, None, [], inputs=inputs)['middleware'])
    inputs['repair'] = True
    wiring = loop._wiring(harness, None, [], inputs=inputs)
    assert any(isinstance(m, AssignmentContext) for m in wiring['middleware'])
    assert not any(isinstance(m, AssignmentContext) for m in wiring['subagents'][0].get('middleware', []))
    with pytest.raises(ValidationError, match='Request-based assignment'):
        AgentConfig.model_validate({**raw,'coordinator_assignment_requires_failure':False})
    requested = AgentConfig.model_validate({**raw,'coordinator_assignment_requires_failure':False,
                                            'subagent_initial_review_required':False})
    assert not requested.subagent_initial_review_required
    with pytest.raises(ValidationError,match='declared string input'):
        AgentConfig.model_validate({**raw,'coordinator_assignment_contract_input':'repair'})
    with pytest.raises(ValidationError, match='must name declared tools'):
        AgentConfig.model_validate({**raw, 'coordinator_assignment_prepare_tool': 'missing'})
    prepared = AgentConfig.model_validate({**raw, 'coordinator_assignment_prepare_tool': 'support__create_refund'})
    assert prepared.coordinator_assignment_prepare_tool == 'support__create_refund'


@pytest.mark.asyncio
async def test_actual_preparation_then_bound_assignment_and_fresh_review_survive_checkpoint():
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import InMemorySaver
    invoked = []

    @tool
    def reference():
        """Read caller contract."""
        invoked.append('reference')
        return 'full contract'

    @tool
    def prepare(profile_id: str):
        """Prepare declared resources; no acceptance runs."""
        invoked.append('prepare:' + profile_id)
        return {'ready': False, 'execution_error': 'existing source failed health', 'checks_executed': False,
                'preparation_report': '/evidence/actual-setup.json'}

    @tool
    def review():
        """Execute actual independent check."""
        invoked.append('review')
        return receipt(True)

    parent = Model(responses=[call('reference','read'),
        call('task','premature',subagent_type='writer',description='must not execute'),
        AIMessage(content='',tool_calls=[{'name':'prepare','id':'setup','args':{'profile_id':'api-contract'}},
                                         {'name':'prepare','id':'duplicate','args':{'profile_id':'api-contract'}}]),
        call('task','handoff',subagent_type='writer',description='invented design'),
        AIMessage(content='Premature finish'),call('review','after'),AIMessage(content='Actual check passed')])
    child = Model(responses=[AIMessage(content='Returned implementation work')])
    stage = AssignmentContext('Original full requirement', ['reference'], ['review'],
        requires_failure=False, routing_retries=2, contract='Exact complete component', prepare_tool='prepare')
    agent = create_deep_agent(model=parent,tools=[reference,prepare,review],
        middleware=[DelegationPrerequisites(['reference'],['review'],final_review_retries=2,
                    initial_review_required=False),stage],
        subagents=[{'name':'writer','description':'Implement','system_prompt':'Preserve content',
                    'model':child,'tools':[]}],checkpointer=InMemorySaver())
    config = {'configurable':{'thread_id':'preparation'}}
    result = await agent.ainvoke({'messages':[{'role':'user','content':'Original full requirement'}]},config)
    assert invoked == ['reference','prepare:api-contract','review']
    assert parent.tool_views[1:4] == [['prepare'],['prepare'],['task']]
    assert 'PREPARE DECLARED RUNTIME' in str(parent.message_views[1])
    assert 'Actual check passed' == result['messages'][-1].content
    text = str(child.message_views[0])
    assert 'Exact complete component' in text and '/evidence/actual-setup.json' in text
    assert 'existing source failed health' in text and 'invented design' not in text
    state = (await agent.aget_state(config)).values
    records = [r for r in state['assignment_receipts'] if r['tool']=='prepare']
    assert len(records)==1 and 'outcomes' not in records[0]
    assert not stage._preparing({'messages':[], 'assignment_receipts':records + [{'tool':'reference','tool_call_id':'read'}]})
    assert 'assignment_receipts' not in result
    assert 'assignment_receipts' not in agent.get_input_jsonschema()['properties']
