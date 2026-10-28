from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
import pytest
from pydantic import Field, ValidationError
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from charter.harness.caller_deadline import CallerDeadline


@pytest.mark.parametrize('value',[None,True,'not a date','2026-10-19T12:00:00'])
def test_invalid_or_naive_deadlines_cannot_silently_become_valid(value):
    with pytest.raises(ValueError): CallerDeadline(value)


def test_dynamic_remaining_time_is_provider_only_and_system_prompt_stays_stable():
    current=[datetime(2026,10,19,12,tzinfo=timezone.utc)]
    deadline=CallerDeadline('2026-10-19T12:20:00Z',now=lambda:current[0])
    original=[HumanMessage(content='Exact original contract')]
    request=ModelRequest(model=FakeMessagesListChatModel(responses=[]),messages=original,
                         state={'messages':original},system_prompt='Standing permissions',tools=[])
    view=deadline._request(request)
    assert '1200 seconds remain' in view.messages[-1].content
    current[0]+=timedelta(seconds=500)
    assert '700 seconds remain' in deadline._request(request).messages[-1].content
    current[0]+=timedelta(seconds=1000)
    assert '0 seconds remain' in deadline._request(request).messages[-1].content
    assert 'never waives fresh review' in view.messages[-1].content
    assert view.system_message == request.system_message
    assert request.messages == original and request.state['messages']==original
    assert deadline.reserved_tokens()>0


def test_declared_opt_in_wiring_reaches_coordinator_and_child_without_changing_limits():
    from charter.config.loader import load_agent
    from charter.config.agent import AgentConfig
    from charter.workflows.loop import Loop
    bundle=load_agent(Path(__file__).parents[1]/'examples/refund-triage')
    raw=bundle.latest.model_dump(by_alias=True)
    with pytest.raises(ValidationError,match='declared string input'):
        AgentConfig.model_validate({**raw,'caller_deadline_input':'missing'})
    raw.update(inputs={**raw['inputs'],'deadline':{'type':'string','default':''}},caller_deadline_input='deadline')
    cfg=AgentConfig.model_validate(raw)
    loop=Loop(cfg,bundle.runtime,tools=SimpleNamespace(langchain_tools=lambda:[]),
              chat_model=lambda _:object(),store_url='postgresql://unused')
    h=SimpleNamespace(wiring={'subagents':[{'name':'writer','tools':[]}]})
    empty=loop._wiring(h,None,[],inputs={'deadline':''})
    assert not any(isinstance(m,CallerDeadline) for m in empty.get('middleware',[]))
    active=loop._wiring(h,None,[],inputs={'deadline':'2026-10-19T12:20:00Z'})
    assert any(isinstance(m,CallerDeadline) for m in active['middleware'])
    assert any(isinstance(m,CallerDeadline) for m in active['subagents'][0]['middleware'])
    assert loop._context_reservation({'deadline':'2026-10-19T12:20:00Z'})>loop._context_reservation({'deadline':''})
    assert bundle.runtime == loop.runtime


@pytest.mark.asyncio
async def test_deadline_survives_assignment_view_and_cannot_replace_fresh_review():
    from deepagents import create_deep_agent
    from charter.harness.assignment_context import AssignmentContext
    from charter.harness.delegation_prerequisites import DelegationPrerequisites
    class Model(FakeMessagesListChatModel):
        views:list=Field(default_factory=list)
        def bind_tools(self,tools,**kw): return self
        def _generate(self,messages,stop=None,run_manager=None,**kw):
            self.views.append(messages)
            return super()._generate(messages,stop=stop,run_manager=run_manager,**kw)
    def call(name,ident,**args):return AIMessage(content='',tool_calls=[{'name':name,'id':ident,'args':args}])
    executed=[]
    @tool
    def reference():
        """Read original contract."""
        return 'contract'
    @tool
    def review():
        """Run actual fresh check."""
        executed.append('review')
        return {'ok':False,'profile':'check','exit_code':1,'report_path':'actual.json','case_summary':{'counts':{'fail':1}}}
    parent=Model(responses=[call('reference','read'),call('task','handoff',subagent_type='writer',description='contract'),
        AIMessage(content='Premature finish'),call('review','after'),AIMessage(content='Check still failed')])
    child=Model(responses=[AIMessage(content='Returned work')])
    deadline=CallerDeadline('2026-10-19T12:20:00Z',now=lambda:datetime(2026,10,19,12,tzinfo=timezone.utc))
    graph=create_deep_agent(model=parent,tools=[reference,review],middleware=[
        DelegationPrerequisites(['reference'],['review'],final_review_retries=2,initial_review_required=False),
        AssignmentContext('original',['reference'],['review'],requires_failure=False),deadline],
        subagents=[{'name':'writer','description':'Implement','system_prompt':'Preserve content','model':child,
                    'tools':[],'middleware':[deadline]}])
    result=await graph.ainvoke({'messages':[{'role':'user','content':'original'}]})
    assert executed==['review'] and result['messages'][-1].content=='Check still failed'
    assert 'ASSIGNMENT STAGE:' in str(parent.views[1])
    assert '1200 seconds remain' in str(parent.views[1]) and '1200 seconds remain' in str(child.views[0])
