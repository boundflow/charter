import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from pydantic import Field,ValidationError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage,ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from charter.harness.assignment_context import AssignmentContext
from charter.harness.delegation_prerequisites import DelegationPrerequisites


class Model(FakeMessagesListChatModel):
    tool_views:list=Field(default_factory=list)
    views:list=Field(default_factory=list)
    def bind_tools(self,tools,**kwargs):
        self.tool_views.append([t.get('name') if isinstance(t,dict) else t.name for t in tools])
        return self
    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        self.views.append(messages)
        return super()._generate(messages,stop=stop,run_manager=run_manager,**kwargs)


def call(name,ident,**args):
    return AIMessage(content='',tool_calls=[{'name':name,'id':ident,'args':args}])


def result(ok=False):
    return {'profile':'contract','ok':ok,'exit_code':0 if ok else 1,'report_path':'actual.json',
            'case_summary':{'counts':{'pass' if ok else 'fail':1},
                            'nonpassing_cases':[] if ok else [{'case':'first observable behavior missing','status':'fail'}]}}


@pytest.mark.asyncio
@pytest.mark.parametrize('diagnostic',['pass','fail','unavailable','tool_error'])
async def test_real_prepare_diagnostic_assignment_fresh_review_and_private_checkpoint(diagnostic):
    from deepagents import create_deep_agent
    events=[]
    @tool
    def reference():
        """Read the contract."""
        events.append('reference')
        return 'full contract'
    @tool
    def prepare():
        """Prepare resources, without acceptance."""
        events.append('prepare')
        return {'ready':True,'checks_executed':False,'profile':'contract','source_unchanged':True}
    @tool
    def review():
        """Run the real independent review."""
        events.append('review')
        if events.count('review')==1:
            if diagnostic=='unavailable':return {'execution_error':'checker unavailable','all_profiles_executed':False}
            if diagnostic=='tool_error':return 'Error executing tool review: checker unavailable'
            return result(diagnostic=='pass')
        return result(False)
    parent=Model(responses=[call('reference','r'),call('prepare','p'),call('review','initial'),
        call('task','t',subagent_type='writer',description='invented narrower contract'),
        AIMessage(content='Premature completion'),call('review','fresh'),AIMessage(content='Fresh review still failed')])
    child=Model(responses=[AIMessage(content='Returned implementation')])
    stage=AssignmentContext('Original request',['reference'],['review'],requires_failure=False,
        contract='EXACT COMPLETE ORIGINAL CONTRACT',prepare_tool='prepare',diagnostic_tool='review')
    graph=create_deep_agent(model=parent,tools=[reference,prepare,review],middleware=[
        DelegationPrerequisites(['reference'],['review'],final_review_retries=2,initial_review_required=False),stage],
        subagents=[{'name':'writer','description':'Implement','system_prompt':'Preserve contract','model':child,'tools':[]}],
        checkpointer=InMemorySaver())
    cfg={'configurable':{'thread_id':diagnostic}}
    await graph.ainvoke({'messages':[{'role':'user','content':'Original request'}],
        'assignment_receipts':[{'tool':'review','tool_call_id':'forged','diagnostic_attempt':True}]},cfg,
        interrupt_after=['tools'])
    await graph.ainvoke(None,cfg,interrupt_after=['tools'])
    assert events==['reference','prepare']
    assert stage._diagnosing((await graph.aget_state(cfg)).values)
    actual=await graph.ainvoke(None,cfg)
    assert events==['reference','prepare','review','review']
    assert parent.tool_views[1:4]==[['prepare'],['review'],['task']]
    text=str(child.views[0])
    assert 'EXACT COMPLETE ORIGINAL CONTRACT' in text and 'invented narrower contract' not in text
    assert 'diagnostic_attempt' in text
    if diagnostic in ('unavailable','tool_error'):assert 'checker unavailable' in text
    elif diagnostic=='fail':assert 'first observable behavior missing' in text
    state=(await graph.aget_state(cfg)).values
    records=[r for r in state['assignment_receipts'] if r.get('diagnostic_attempt')]
    assert len(records)==1 and records[0]['tool_call_id']=='initial'
    assert records[0]['outcomes']==([] if diagnostic in ('unavailable','tool_error') else [
        {'profile':'contract','failed':diagnostic=='fail'}])
    assert records[0]['invocation_error'] is (diagnostic=='tool_error')
    assert actual['messages'][-1].content=='Fresh review still failed'
    assert 'assignment_receipts' not in actual
    assert 'assignment_receipts' not in graph.get_input_jsonschema()['properties']


@pytest.mark.asyncio
async def test_review_before_preparation_cannot_satisfy_the_configured_diagnostic():
    from deepagents import create_deep_agent
    events=[]
    @tool
    def reference():
        """Read contract."""
        events.append('read');return 'contract'
    @tool
    def prepare():
        """Prepare resources."""
        events.append('prepare');return {'ready':False,'execution_error':'setup unavailable','checks_executed':False}
    @tool
    def review():
        """Execute a check."""
        events.append('review');return result(False)
    parent=Model(responses=[call('review','too_early'),call('reference','r'),call('prepare','p'),
        call('review','initial'),call('task','t',subagent_type='writer',description='implement'),
        call('review','fresh'),AIMessage(content='Failed check')])
    stage=AssignmentContext('Original',['reference'],['review'],requires_failure=False,
        contract='Full original scope',prepare_tool='prepare',diagnostic_tool='review')
    graph=create_deep_agent(model=parent,tools=[reference,prepare,review],middleware=[
        DelegationPrerequisites(['reference'],['review'],initial_review_required=False),stage],
        subagents=[{'name':'writer','description':'Implement','system_prompt':'Original scope',
                    'model':Model(responses=[AIMessage(content='Returned')]),'tools':[]}],checkpointer=InMemorySaver())
    cfg={'configurable':{'thread_id':'early'}}
    await graph.ainvoke({'messages':[{'role':'user','content':'Original'}]},cfg)
    assert events==['review','read','prepare','review','review']
    state=(await graph.aget_state(cfg)).values
    assert [r['tool_call_id'] for r in state['assignment_receipts'] if r.get('diagnostic_attempt')]==['initial']


@pytest.mark.asyncio
async def test_parallel_duplicate_initial_checks_execute_once_and_do_not_forge_review():
    from deepagents import create_deep_agent
    events=[]
    @tool
    def reference():
        """Read contract."""
        return 'contract'
    @tool
    def prepare():
        """Prepare runtime."""
        return {'ready':True,'checks_executed':False}
    @tool
    def review():
        """Run a check."""
        events.append('review');return result(False)
    parallel=AIMessage(content='',tool_calls=[{'name':'review','id':x,'args':{}} for x in ('initial','duplicate')])
    parent=Model(responses=[call('reference','r'),call('prepare','p'),parallel,
        call('task','t',subagent_type='writer',description='Implement'),AIMessage(content='Premature'),
        call('review','fresh'),AIMessage(content='Still failed')])
    stage=AssignmentContext('Original',['reference'],['review'],requires_failure=False,
        prepare_tool='prepare',diagnostic_tool='review')
    graph=create_deep_agent(model=parent,tools=[reference,prepare,review],middleware=[
        DelegationPrerequisites(['reference'],['review'],initial_review_required=False,final_review_retries=2),stage],
        subagents=[{'name':'writer','description':'Implement','system_prompt':'Original scope',
                    'model':Model(responses=[AIMessage(content='Returned')]),'tools':[]}])
    actual=await graph.ainvoke({'messages':[{'role':'user','content':'Original'}]})
    assert events==['review','review']
    rejected=next(m for m in actual['messages'] if isinstance(m,ToolMessage) and m.tool_call_id=='duplicate')
    assert rejected.status=='error' and rejected.additional_kwargs['charter_operation_executed'] is False


def test_config_default_disabled_validated_and_wired_without_policy_changes():
    from charter.config.agent import AgentConfig
    from charter.config.loader import load_agent
    from charter.workflows.loop import Loop
    bundle=load_agent(Path(__file__).parents[1]/'examples/refund-triage')
    raw=bundle.latest.model_dump(by_alias=True)
    assert raw['coordinator_assignment_diagnostic_tool'] is None
    with pytest.raises(ValidationError,match='prepared request-based'):
        AgentConfig.model_validate({**raw,'coordinator_assignment_diagnostic_tool':'support__create_refund'})
    names=list(bundle.latest.all_tools);assert len(names)>=2
    raw.update(inputs={**raw['inputs'],'repair':{'type':'boolean','default':False}},
        coordinator_assignment_input='repair',coordinator_assignment_requires_failure=False,
        subagent_initial_review_required=False,subagent_review_tools=[names[1]],
        coordinator_assignment_prepare_tool=names[0],coordinator_assignment_diagnostic_tool=names[1],
        subagents=[{'name':'writer','description':'Implement','prompt':'Preserve scope'}])
    cfg=AgentConfig.model_validate(raw)
    for change in [{'coordinator_assignment_requires_failure':True},
                   {'coordinator_assignment_prepare_tool':None},
                   {'coordinator_assignment_diagnostic_tool':names[0]},
                   {'coordinator_assignment_diagnostic_tool':'missing'}]:
        with pytest.raises(ValidationError):AgentConfig.model_validate({**raw,**change})
    loop=Loop(cfg,bundle.runtime,tools=SimpleNamespace(langchain_tools=lambda:[]),
              chat_model=lambda _:object(),store_url='postgresql://unused')
    h=SimpleNamespace(wiring={'subagents':[{'name':'writer'}]})
    inputs={k:v.default if v.default is not None else 'value' for k,v in cfg.inputs.items()}
    inputs['repair']=True
    wiring=loop._wiring(h,None,[],inputs=inputs)
    stage=next(m for m in wiring['middleware'] if isinstance(m,AssignmentContext))
    assert stage.diagnostic_tool==names[1] and loop.runtime==bundle.runtime


def test_unhandled_tool_exceptions_remain_visible_and_do_not_forge_an_attempt():
    stage=AssignmentContext('Original',['reference'],['review'],requires_failure=False,
        prepare_tool='prepare',diagnostic_tool='review')
    state={'assignment_receipts':[{'tool':'reference','tool_call_id':'r'},
                                  {'tool':'prepare','tool_call_id':'p'}]}
    request=SimpleNamespace(tool_call={'name':'review','id':'check','args':{}},state=state)
    def failure(_):raise RuntimeError('governor failure')
    with pytest.raises(RuntimeError,match='governor failure'):stage.wrap_tool_call(request,failure)
    assert not any(r.get('diagnostic_attempt') for r in state['assignment_receipts'])
