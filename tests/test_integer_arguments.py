from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.tools import tool
from pydantic import StrictInt, Field
from typing import Annotated
from langgraph.types import Command

from charter.harness.integer_arguments import CanonicalIntegerArguments, normalize_arguments, ANNOTATION


def test_only_declared_unambiguous_canonical_integer_representations_change():
    schema = {'properties': {'offset': {'type':'integer'},
        'end': {'anyOf':[{'type':'integer'}, {'type':'null'}]},
        'label': {'type':'string'}, 'ambiguous': {'anyOf':[{'type':'integer'}, {'type':'string'}]}}}
    args = {'offset':'0','end':'701','label':'0003','ambiguous':'3','unknown':'4'}
    normalized, fields = normalize_arguments(args, schema)
    assert normalized == {**args, 'offset':0,'end':701}
    assert fields == ['end','offset']
    assert args['offset'] == '0' and args['end'] == '701'
    for value in [True, 2.0, '01', '-0', '+1', '1.0', '1e2', ' 1', '1 ', '1,', '١', '9'*17, None]:
        assert normalize_arguments({'offset':value},schema) == ({'offset':value},[])
    assert normalize_arguments({'offset':'-1'},schema) == ({'offset':-1},['offset'])


@dataclass
class Request:
    tool_call: dict
    tool: object
    def override(self, **updates):
        return replace(self, **updates)


def test_unknown_tools_and_command_receipts_keep_original_guards_and_content():
    middleware = CanonicalIntegerArguments()
    request = Request({'id':'one','name':'bounded','args':{'offset':'2'}},None)
    assert middleware._request(request) == (request,[])
    request.tool = SimpleNamespace(args_schema={'properties':{'offset':{'type':'integer'}}})
    content = '{"ok":false,"execution_error":"unavailable"}'
    message = ToolMessage(content=content,tool_call_id='one',status='error',additional_kwargs={'prior':'retained'})
    command = Command(update={'messages':[message],'existing_private_receipts':[{'actual':'event'}]})
    result = middleware.wrap_tool_call(request,lambda r: command if r.tool_call['args']=={'offset':2} else pytest.fail('Not normalized'))
    actual = result.update['messages'][0]
    assert actual.content == content and actual.status == 'error'
    assert actual.additional_kwargs == {'prior':'retained',ANNOTATION:['offset']}
    assert result.update['existing_private_receipts'] == command.update['existing_private_receipts']
    assert ANNOTATION not in message.additional_kwargs and request.tool_call['args']['offset']=='2'


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_real_graph_still_validates_bounds_boolean_and_string_guards_after_transport():
    from deepagents import create_deep_agent
    calls=[]
    @tool
    def bounded(limit: Annotated[StrictInt,Field(ge=1,le=3)], guard: str, end: StrictInt|None=None):
        """Read a bounded range, preserving the caller guard."""
        calls.append((limit,guard,end))
        return 'actual bounded read'
    messages=[]
    for ident,limit in [('outside','0'),('boolean',True),('valid','2')]:
        messages.append(AIMessage(content='',tool_calls=[{'name':'bounded','id':ident,
            'args':{'limit':limit,'guard':'0003','end':'3'}}]))
    messages.append(AIMessage(content='Returned measured outcome'))
    agent=create_deep_agent(model=Model(responses=messages),tools=[bounded],middleware=[CanonicalIntegerArguments()])
    result=await agent.ainvoke({'messages':[{'role':'user','content':'Read within bounds'}]})
    assert calls==[(2,'0003',3)]
    receipts=[m for m in result['messages'] if isinstance(m,ToolMessage)]
    assert [m.status for m in receipts]==['error','error','success']
    assert receipts[-1].additional_kwargs[ANNOTATION]==['end','limit']


def test_configuration_is_strictly_opt_in_and_wires_both_coordinator_and_specialist():
    from pathlib import Path
    from charter.config.loader import load_agent
    from charter.config.agent import AgentConfig
    from charter.workflows.loop import Loop
    from pydantic import ValidationError
    bundle=load_agent(Path(__file__).parents[1]/'examples/refund-triage')
    raw=bundle.latest.model_dump(by_alias=True)
    assert not bundle.latest.normalize_tool_integer_arguments
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({**raw,'normalize_tool_integer_arguments':'true'})
    raw.update(normalize_tool_integer_arguments=True,
               subagents=[{'name':'writer','description':'Write','prompt':'Implement'}])
    config=AgentConfig.model_validate(raw)
    loop=Loop(config,bundle.runtime,tools=SimpleNamespace(langchain_tools=lambda:[]),chat_model=lambda _:object(),store_url='postgresql://unused')
    wiring=loop._wiring(SimpleNamespace(wiring={'subagents':[{'name':'writer'}]}),None,[])
    assert any(isinstance(m,CanonicalIntegerArguments) for m in wiring['middleware'])
    assert any(isinstance(m,CanonicalIntegerArguments) for m in wiring['subagents'][0]['middleware'])
