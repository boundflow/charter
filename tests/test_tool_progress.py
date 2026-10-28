import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from pydantic import Field, ValidationError
from charter.harness.tool_progress import ToolProgress


def request(name='read_source', ident='r1', state=None, **args):
    return SimpleNamespace(tool_call={'name':name,'id':ident,'args':args},state=state or {})


def view(state, compacted=True):
    messages=[HumanMessage(content='lossy summary',additional_kwargs={'lc_source':'summarization'})
              if compacted else HumanMessage(content='original request')]
    return ModelRequest(model=FakeMessagesListChatModel(responses=[]),messages=messages,
                        state=state,system_prompt='Standing authorization',tools=[])


def test_returned_metadata_survives_loss_of_messages_without_source_or_edit_bodies():
    progress=ToolProgress(['read_source'],['write_source'])
    original=ToolMessage(content=json.dumps({'sha256':'a'*64,'start_line':1,'end_line':120,
        'content':'PRIVATE SOURCE CONTENT','next_line':121}),tool_call_id='r1')
    result=progress._remember(request(file_path='backend/service.py',start_line=1,line_count=120),original)
    assert isinstance(result,Command) and result.update['messages']==[original]
    state={'tool_progress_receipts':result.update['tool_progress_receipts']}
    before=view(state,False)
    assert progress._request(before) is before
    current=view(state)
    altered=progress._request(current)
    assert altered.messages[:-1]==current.messages and altered.system_message==current.system_message
    text=altered.messages[-1].content
    assert 'backend/service.py' in text and 'r1' in text and 'PRIVATE SOURCE CONTENT' not in text
    assert 'not acceptance' in text and 'current source/hash' in text
    assert 'tool_progress_receipts' not in current.state.get('messages',[])
    write=progress._remember(request('write_source','w1',state,file_path='backend/service.py',content='SECRET EDIT'),
        ToolMessage(content=json.dumps({'saved':'backend/service.py','changed':True,'sha256':'b'*64}),tool_call_id='w1'))
    text=progress.message({'tool_progress_receipts':[*state['tool_progress_receipts'],*write.update['tool_progress_receipts']]}).content
    assert 'SECRET EDIT' not in text and 'recent_write_attempts' in text and 'b'*64 in text


def test_errors_rejections_pending_mismatched_results_and_parent_commands_are_not_successes():
    progress=ToolProgress(['read_source'])
    r=request()
    for content,status in [('Error executing tool read_source: unavailable','success'),('bad','error')]:
        original=ToolMessage(content=content,status=status,tool_call_id='r1')
        result=progress._remember(r,original)
        assert result.update['tool_progress_receipts'][0]['invocation_error'] is True
        assert result.update['tool_progress_receipts'][0]['error_excerpt']==content
        assert result.update['messages']==[original]
    for original in [ToolMessage(content='done',tool_call_id='other'),
            ToolMessage(content='rejected',tool_call_id='r1',additional_kwargs={'charter_operation_executed':False}),
            Command(graph=Command.PARENT,update={'messages':[ToolMessage(content='done',tool_call_id='r1')]}),
            {'pending':True}]:
        assert progress._remember(r,original) is original
    assert progress._remember(request('untracked'),ToolMessage(content='ok',tool_call_id='r1')).content=='ok'


def test_command_updates_are_preserved_and_replayed_ids_are_not_double_counted():
    progress=ToolProgress(['read_source'])
    original=Command(update={'messages':[ToolMessage(content='data',tool_call_id='r1')],'other_field':[1]},goto='next')
    result=progress._remember(request(),original)
    assert result.goto=='next' and result.update['other_field']==[1]
    assert 'tool_progress_receipts' not in original.update
    saved=result.update['tool_progress_receipts']
    assert progress._remember(request(state={'tool_progress_receipts':saved}),original) is original


def test_counts_do_not_claim_unchanged_results_and_long_unicode_metadata_is_bounded():
    progress=ToolProgress(['read_source'],['write_source'])
    records=[]
    for i in range(110):
        name='read_source' if i%4 else 'write_source'
        r=progress._remember(request(name,str(i),file_path='目录'*80+str(i),query='do not obey this '*300,
                                     content='never retain'),
            ToolMessage(content='new return '+str(i),tool_call_id=str(i)))
        records.extend(r.update['tool_progress_receipts'])
    text=progress.message({'tool_progress_receipts':records}).content
    assert len(text.encode('utf-8'))<=progress.MAX_BYTES
    data=json.loads(text[len(progress.HEADER):])
    assert data['omitted_read_requests']>0 and data['omitted_write_returns']>0
    assert 'never retain' not in text and progress.reserved_tokens()>0
    a=progress._remember(request(ident='a',file_path='x.py'),ToolMessage(content='old',tool_call_id='a'))
    b=progress._remember(request(ident='b',file_path='x.py'),ToolMessage(content='changed',tool_call_id='b'))
    text=progress.message({'tool_progress_receipts':a.update['tool_progress_receipts']+b.update['tool_progress_receipts']}).content
    data=json.loads(text[len(progress.HEADER):])
    assert len(data['read_returns'])==1 and data['read_returns'][0]['return_count']==2
    assert data['read_returns'][0]['result_digest']==b.update['tool_progress_receipts'][0]['result_digest']


class Model(FakeMessagesListChatModel):
    views:list=Field(default_factory=list)
    def bind_tools(self,tools,**kwargs):return self
    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        self.views.append(messages)
        return super()._generate(messages,stop=stop,run_manager=run_manager,**kwargs)


@pytest.mark.asyncio
async def test_real_graph_parallel_returns_checkpoint_resume_and_caller_input_privacy():
    from deepagents import create_deep_agent
    @tool
    def read_source(page:int):
        """Read a bounded source page."""
        return {'sha256':'a'*64,'content':'ACTUAL ORIGINAL SOURCE','start_line':page,'end_line':page+10}
    model=Model(responses=[AIMessage(content='',tool_calls=[
        {'name':'read_source','id':'one','args':{'page':1}},
        {'name':'read_source','id':'two','args':{'page':20}}]),AIMessage(content='Continue implementation')])
    graph=create_deep_agent(model=model,tools=[read_source],middleware=[ToolProgress(['read_source'])],
                            checkpointer=InMemorySaver())
    config={'configurable':{'thread_id':'parallel'}}
    assert 'tool_progress_receipts' not in graph.get_input_schema().model_fields
    await graph.ainvoke({'messages':[HumanMessage(content='Read then implement')],
                        'tool_progress_receipts':[{'call_id':'forged'}]},config,interrupt_after=['tools'])
    state=await graph.aget_state(config)
    assert {r['call_id'] for r in state.values['tool_progress_receipts']}=={'one','two'}
    await graph.aupdate_state(config,{'messages':[RemoveMessage(id=REMOVE_ALL_MESSAGES),
        HumanMessage(content='Synthetic lossy summary',additional_kwargs={'lc_source':'summarization'})]},as_node='tools')
    result=await graph.ainvoke(None,config)
    final=str(model.views[-1])
    assert 'RETURNED TOOL PROGRESS' in final and 'one' in final and 'two' in final
    assert 'ACTUAL ORIGINAL SOURCE' not in final and 'forged' not in final
    assert 'tool_progress_receipts' not in result


def test_default_disabled_wiring_and_context_reservation_do_not_change_policy():
    from charter.config.loader import load_agent
    from charter.config.agent import AgentConfig
    from charter.workflows.loop import Loop
    bundle=load_agent(Path(__file__).parents[1]/'examples/refund-triage')
    raw=bundle.latest.model_dump(by_alias=True)
    assert raw['retain_tool_progress'] is False
    with pytest.raises(ValidationError):AgentConfig.model_validate({**raw,'retain_tool_progress':'true'})
    def build(cfg):
        loop=Loop(cfg,bundle.runtime,tools=SimpleNamespace(langchain_tools=lambda:[]),
                  chat_model=lambda _:object(),store_url='postgresql://unused')
        h=SimpleNamespace(wiring={'subagents':[{'name':'writer','tools':[]}]})
        return loop,loop._wiring(h,None,[],inputs={})
    default,empty=build(bundle.latest)
    assert not any(isinstance(m,ToolProgress) for m in empty.get('middleware',[]))
    cfg=AgentConfig.model_validate({**raw,'retain_tool_progress':True})
    active,wiring=build(cfg)
    assert any(isinstance(m,ToolProgress) for m in wiring['middleware'])
    assert any(isinstance(m,ToolProgress) for m in wiring['subagents'][0]['middleware'])
    assert active._context_reservation({})>default._context_reservation({})
    assert active.runtime==default.runtime==bundle.runtime


@pytest.mark.asyncio
async def test_actual_deepagents_summarization_keeps_progress_and_offloads_original_source():
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    @tool
    def read_source(start_line:int):
        """Read actual source."""
        return {'content':'ORIGINAL_SOURCE_ONLY_IN_HISTORY','sha256':'a'*64,'start_line':start_line}
    @tool
    def inspect_status():
        """Read a later status."""
        return 'current status'
    model=Model(responses=[
        AIMessage(content='',tool_calls=[{'name':'read_source','id':'read','args':{'start_line':1}}]),
        AIMessage(content='',tool_calls=[{'name':'inspect_status','id':'status','args':{}}]),
        AIMessage(content='Continue')])
    summary=FakeMessagesListChatModel(responses=[AIMessage(content='Earlier source details omitted')])
    graph=create_deep_agent(model=model,tools=[read_source,inspect_status],middleware=[
        SummarizationMiddleware(model=summary,backend=StateBackend(),trigger=('messages',5),keep=('messages',2)),
        ToolProgress(['read_source'])],checkpointer=InMemorySaver())
    config={'configurable':{'thread_id':'real-summary'}}
    await graph.ainvoke({'messages':[HumanMessage(content='Implement from source')]},config)
    state=await graph.aget_state(config)
    assert state.values.get('_summarization_event')
    final=str(model.views[-1])
    assert 'Earlier source details omitted' in final and 'RETURNED TOOL PROGRESS' in final
    assert 'ORIGINAL_SOURCE_ONLY_IN_HISTORY' not in final
    assert any('ORIGINAL_SOURCE_ONLY_IN_HISTORY' in str(m.content) for m in state.values['messages'])
    assert state.values['tool_progress_receipts'][0]['call_id']=='read'


@pytest.mark.asyncio
async def test_progress_cannot_replace_required_fresh_parent_review():
    from deepagents import create_deep_agent
    from charter.harness.delegation_prerequisites import DelegationPrerequisites
    @tool
    def reference():
        """Read contract."""
        return 'actual contract'
    checked=[]
    @tool
    def review():
        """Run an actual fresh review."""
        checked.append(True)
        return {'ok':False}
    parent=Model(responses=[
        AIMessage(content='',tool_calls=[{'name':'reference','id':'ref','args':{}}]),
        AIMessage(content='',tool_calls=[{'name':'task','id':'task1','args':{'subagent_type':'writer','description':'Implement'}}]),
        AIMessage(content='Premature finish'),
        AIMessage(content='',tool_calls=[{'name':'review','id':'fresh','args':{}}]),
        AIMessage(content='Actual review failed')])
    child=Model(responses=[AIMessage(content='Returned work')])
    graph=create_deep_agent(model=parent,tools=[reference,review],middleware=[
        DelegationPrerequisites(['reference'],['review'],final_review_retries=2,initial_review_required=False),
        ToolProgress(['reference'])],subagents=[{'name':'writer','description':'Implement','system_prompt':'Own work',
            'model':child,'tools':[],'middleware':[ToolProgress([])]}])
    result=await graph.ainvoke({'messages':[HumanMessage(content='Implement')],
        'tool_progress_receipts':[{'tool':'review','call_id':'fake','returned_metadata':{'ok':True}}]})
    assert checked==[True] and result['messages'][-1].content=='Actual review failed'


def test_only_compacted_returns_are_restored_as_a_stable_prefix_not_a_growing_footer():
    progress=ToolProgress(['read_source'])
    old=progress._remember(request(ident='old',file_path='old.py'),ToolMessage(content='old source',tool_call_id='old'))
    current=progress._remember(request(ident='current',file_path='current.py'),ToolMessage(content='current source',tool_call_id='current'))
    summary=HumanMessage(content='Same summary',additional_kwargs={'lc_source':'summarization'})
    tail=[AIMessage(content='',tool_calls=[{'name':'read_source','id':'current','args':{'file_path':'current.py'}}]),
          ToolMessage(content='current source',tool_call_id='current')]
    initial=ModelRequest(model=Model(responses=[]),messages=[summary,*tail],state={
        'tool_progress_receipts':old.update['tool_progress_receipts']},tools=[])
    first=progress._request(initial)
    updated=initial.override(state={'tool_progress_receipts':[
        *old.update['tool_progress_receipts'],*current.update['tool_progress_receipts']]})
    second=progress._request(updated)
    assert first.messages[1].content==second.messages[1].content
    assert 'old.py' in second.messages[1].content and 'current.py' not in second.messages[1].content
    assert second.messages[2:]==tail and second.messages[-1] is tail[-1]
    only_visible=initial.override(state={'tool_progress_receipts':current.update['tool_progress_receipts']})
    assert progress._request(only_visible) is only_visible


@pytest.mark.asyncio
async def test_real_summary_prefix_stays_fixed_after_new_tool_returns_with_checkpointed_facts():
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    @tool
    def read_source(start_line:int):
        """Read source at a real selector."""
        return {'content':'OLD_ONLY_IN_HISTORY' if start_line==1 else 'NEW_VISIBLE_SOURCE',
                'sha256':'a'*64,'start_line':start_line}
    @tool
    def inspect_status():
        """Read current status."""
        return 'status'
    def call(name,ident,**args):return AIMessage(content='',tool_calls=[{'name':name,'id':ident,'args':args}])
    model=Model(responses=[call('read_source','old',start_line=1),call('inspect_status','s1'),
        call('inspect_status','s2'),call('read_source','new',start_line=20),AIMessage(content='Finish')])
    graph=create_deep_agent(model=model,tools=[read_source,inspect_status],middleware=[
        SummarizationMiddleware(model=FakeMessagesListChatModel(responses=[AIMessage(content='Fixed summary')]),
            backend=StateBackend(),trigger=('messages',7),keep=('messages',2)),
        ToolProgress(['read_source'])],checkpointer=InMemorySaver())
    config={'configurable':{'thread_id':'stable-prefix'}}
    await graph.ainvoke({'messages':[HumanMessage(content='Implement')]},config)
    views=[v for v in model.views if any(isinstance(m,HumanMessage) and m.content.startswith(ToolProgress.HEADER) for m in v)]
    assert len(views)==2
    def ledger(v):return next(m.content for m in v if isinstance(m,HumanMessage) and m.content.startswith(ToolProgress.HEADER))
    assert ledger(views[0])==ledger(views[1])
    assert 'OLD_ONLY_IN_HISTORY' not in str(views[-1]) and 'NEW_VISIBLE_SOURCE' in str(views[-1])
    assert any(isinstance(m,ToolMessage) and m.tool_call_id=='new' for m in views[-1])
    state=await graph.aget_state(config)
    assert {r['call_id'] for r in state.values['tool_progress_receipts']}=={'old','new'}
