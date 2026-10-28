"""Per-agent removal of unwanted built-in tool alternatives.

This prevents two competing filesystem interfaces in agents with dedicated project
tools. It is a tool-routing constraint, not an operating-system security sandbox.
"""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


class BuiltinToolExclusions(AgentMiddleware):
    def __init__(self, names, hints=None):
        self.names = frozenset(names)
        self.hints = hints or {}

    def _request(self, request):
        tools = []
        for tool in request.tools:
            name = tool.get('name') if isinstance(tool, dict) else tool.name
            if name in self.names:
                continue
            hint = self.hints.get(name)
            if hint:
                description = hint + '\n' + (tool.get('description', '') if isinstance(tool, dict) else tool.description)
                tool = {**tool, 'description': description} if isinstance(tool, dict) else tool.model_copy(update={'description': description})
            tools.append(tool)
        return request.override(tools=tools)

    def wrap_model_call(self, request, handler):
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request(request))

    def _rejection(self, request):
        call = request.tool_call
        if call['name'] in self.names:
            return ToolMessage(content=f"{call['name']} is disabled for this agent. Use its declared project tools or delegate to the assigned specialist.",
                name=call['name'], tool_call_id=call['id'], status='error')
        return None

    def wrap_tool_call(self, request, handler):
        return self._rejection(request) or self._hint(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._rejection(request) or self._hint(request, await handler(request))

    def _hint(self, request, result):
        hint = self.hints.get(request.tool_call['name'])
        if hint and isinstance(result, ToolMessage) and isinstance(result.content, str) and (
                result.status == 'error' or result.content.startswith('Error:')):
            return result.model_copy(update={'content': result.content + '\nTool routing: ' + hint})
        return result
