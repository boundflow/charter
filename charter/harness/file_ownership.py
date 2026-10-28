"""Opt-in file ownership for inline tasks using dedicated path-based tools.

This coordinates declared tools within one parent run. It is not an OS sandbox;
unlisted tools, subprocess side effects and other runs need separate controls.
"""
from contextvars import ContextVar
from copy import deepcopy
import os
from threading import RLock

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage, SystemMessage
import json


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Use nonempty relative file paths')
    value = value.replace('\\', '/')
    if ':' in value or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('Use exact relative files without traversal or drive letters')
    return os.path.normcase(value).replace('\\', '/')


def _error(request, text):
    call = request.tool_call
    return ToolMessage(name=call['name'], tool_call_id=call['id'], status='error', content=text)


class FileOwnership(AgentMiddleware):
    def __init__(self, write_tools, max_files, subagents):
        self.write_tools = write_tools
        self.max_files = max_files
        self.subagents = frozenset(subagents)
        self.scope = ContextVar('inline_task_owned_files', default=None)
        self.display_scope = ContextVar('inline_task_owned_file_spellings', default=None)
        self.active = set()
        self.lock = RLock()

    def _model_request(self, request):
        tools = []
        for tool in request.tools:
            name = tool.get('name') if isinstance(tool, dict) else tool.name
            if name == 'task':
                if isinstance(tool, dict):
                    tool = deepcopy(tool)
                    schema = tool.setdefault('parameters', {})
                else:
                    schema = deepcopy(tool.args_schema if isinstance(tool.args_schema, dict)
                                      else tool.get_input_schema().model_json_schema())
                    tool = tool.model_copy(update={'args_schema': schema})
                schema.setdefault('properties', {})['owned_files'] = {
                    'type': 'array', 'items': {'type': 'string'}, 'maxItems': self.max_files,
                    'uniqueItems': True,
                    'description': 'Exact relative files this specialist may write. Empty means read-only. Concurrent tasks must have disjoint ownership.'}
                specialist = schema['properties'].setdefault('subagent_type', {'type': 'string'})
                specialist['enum'] = sorted(self.subagents)
                if specialist.get('default') not in self.subagents:
                    specialist.pop('default', None)
                schema['required'] = list(dict.fromkeys([*schema.get('required', []), 'owned_files']))
            tools.append(tool)
        return request.override(tools=tools)

    def wrap_model_call(self, request, handler):
        return handler(self._model_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._model_request(request))

    def _claim(self, request):
        args = request.tool_call.get('args', {})
        paths = args.get('owned_files')
        if args.get('subagent_type') not in self.subagents:
            raise ValueError('Choose a configured specialist with file ownership enforcement')
        if not isinstance(paths, list) or len(paths) > self.max_files:
            raise ValueError(f'task requires owned_files: an array of at most {self.max_files} exact paths; [] means read-only')
        owned = frozenset(_path(path) for path in paths)
        if len(owned) != len(paths):
            raise ValueError('owned_files contains duplicate paths')
        if self.scope.get() is not None:
            raise ValueError('Nested delegation cannot replace the current file ownership')
        description = args.get('description')
        if not isinstance(description, str):
            raise ValueError('task description must be a string')
        with self.lock:
            if overlap := self.active.intersection(owned):
                raise ValueError('Files already owned by a running task: ' + ', '.join(sorted(overlap)))
            self.active.update(owned)
        stripped = {key: value for key, value in args.items() if key != 'owned_files'}
        stripped['description'] = description + '\nEnforced write ownership: ' + ', '.join(paths) + (
            '. All other files are read-only. Report needed changes outside this assignment to the coordinator.')
        return owned, tuple(path.replace('\\', '/') for path in paths), request.override(
            tool_call={**request.tool_call, 'args': stripped})

    def _release(self, owned):
        with self.lock:
            self.active.difference_update(owned)

    def wrap_tool_call(self, request, handler):
        if request.tool_call['name'] != 'task':
            return handler(request)
        try:
            owned, displayed, scoped = self._claim(request)
        except ValueError as error:
            return _error(request, str(error))
        token = self.scope.set(owned)
        display_token = self.display_scope.set(displayed)
        try:
            return handler(scoped)
        finally:
            self.scope.reset(token)
            self.display_scope.reset(display_token)
            self._release(owned)

    async def awrap_tool_call(self, request, handler):
        if request.tool_call['name'] != 'task':
            return await handler(request)
        try:
            owned, displayed, scoped = self._claim(request)
        except ValueError as error:
            return _error(request, str(error))
        token = self.scope.set(owned)
        display_token = self.display_scope.set(displayed)
        try:
            return await handler(scoped)
        finally:
            self.scope.reset(token)
            self.display_scope.reset(display_token)
            self._release(owned)


class OwnedWrites(AgentMiddleware):
    def __init__(self, ownership):
        self.ownership = ownership

    def _model_request(self, request):
        scope = self.ownership.scope.get()
        if scope is None:
            return request
        # Task text can be compacted away. Rebuild scope from the live claim,
        # never from a generated summary or a specialist's proposed filenames.
        policy = ('\n\nCURRENT SPECIALIST WRITE SCOPE (enforced runtime claim): '
                  + json.dumps(list(self.ownership.display_scope.get() or sorted(scope)), ensure_ascii=False)
                  + '. All other paths are read-only, including alternate filenames for the same feature. '
                  'Continue from actual completed edits and tool feedback; a historical plan does not '
                  'change this assignment. Resolve interfaces from current source and checker definitions. '
                  'After implementing, run relevant finite checks. If a necessary change is outside this '
                  'scope, return its exact path and reason to the coordinator instead of trying other '
                  'destinations or restarting historical planning. Report unexecuted checks as unexecuted. '
                  'This scope reminder grants no additional permission and claims no successful edit or check.')
        system = request.system_message
        if system is None:
            system = SystemMessage(content=policy)
        else:
            content = system.content + policy if isinstance(system.content, str) else [
                *system.content, {'type': 'text', 'text': policy}]
            system = system.model_copy(update={'content': content})
        return request.override(system_message=system)

    def wrap_model_call(self, request, handler):
        return handler(self._model_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._model_request(request))

    def _rejection(self, request):
        call = request.tool_call
        fields = self.ownership.write_tools.get(call['name'])
        if fields is None:
            return None
        scope = self.ownership.scope.get()
        arguments = call.get('args', {})
        paths = [arguments[field] for field in fields if arguments.get(field) not in (None, '')]
        if not paths:
            return _error(request, 'Write has not executed: no configured destination argument was supplied. '
                          'Use the tool schema; expected a nonempty path in one of: ' + ', '.join(fields)
                          + '. Do not invent argument names. This is not an ownership denial; '
                          'the declared owned_files are unchanged. Complete required diagnostic reads '
                          'first if write tools are not yet offered.')
        try:
            normalized = {_path(path) for path in paths}
        except ValueError as error:
            return _error(request, str(error))
        if len(normalized) != 1:
            return _error(request, 'Write has not executed: conflicting destination arguments were supplied. '
                          'Use the tool schema and one exact destination; owned_files are unchanged.')
        if scope is None or not normalized.issubset(scope):
            return _error(request, 'Write refused: this path is outside the specialist\'s explicit owned_files. '
                          'Current owned_files: ' + json.dumps(list(self.ownership.display_scope.get()
                              or (sorted(scope) if scope is not None else [])))
                          + '. Report the exact needed path to the coordinator; do not try alternate filenames '
                          'or another tool to bypass ownership. Verify completed owned edits and return the gap.')
        return None

    def wrap_tool_call(self, request, handler):
        return self._rejection(request) or handler(request)

    async def awrap_tool_call(self, request, handler):
        return self._rejection(request) or await handler(request)
