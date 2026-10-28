"""Opt-in canonical decimal transport for fields declared integer-only.

This changes representation before normal tool validation, never bounds, tool
availability, authorization, unknown fields, required inputs or result content.
"""
from dataclasses import replace
import logging
import re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

log = logging.getLogger(__name__)
ANNOTATION = 'charter_normalized_integer_arguments'


def _integer_only(schema):
    if not isinstance(schema, dict):
        return False
    kind = schema.get('type')
    if kind == 'integer':
        return True
    if isinstance(kind, list):
        return 'integer' in kind and set(kind) <= {'integer', 'null'}
    branches = schema.get('anyOf', schema.get('oneOf', []))
    kinds = [b.get('type') for b in branches if isinstance(b, dict)]
    return bool(branches) and len(kinds) == len(branches) and 'integer' in kinds and all(
        kind in ('integer', 'null') for kind in kinds)


def normalize_arguments(arguments, schema):
    """Normalize only flat, unambiguous integer fields; leave every other input intact."""
    result, changed = dict(arguments), []
    for field, definition in schema.get('properties', {}).items():
        value = arguments.get(field)
        if (_integer_only(definition) and isinstance(value, str) and len(value) <= 16
                and re.fullmatch(r'(?:0|-?[1-9][0-9]*)', value)):
            result[field] = int(value)
            changed.append(field)
    return result, sorted(changed)


class CanonicalIntegerArguments(AgentMiddleware):
    def _request(self, request):
        if request.tool is None:
            return request, []  # Unregistered tools retain ordinary refusal.
        schema = request.tool.args_schema
        if not isinstance(schema, dict):
            schema = request.tool.get_input_schema().model_json_schema()
        arguments, fields = normalize_arguments(request.tool_call.get('args', {}), schema)
        if not fields:
            return request, fields
        log.warning('Normalized canonical decimal arguments for %s: %s; normal validation remains required',
                    request.tool_call['name'], ', '.join(fields))
        return request.override(tool_call={**request.tool_call, 'args': arguments}), fields

    def _annotate(self, result, fields, ident):
        if not fields:
            return result
        def message(value):
            if isinstance(value, ToolMessage) and value.tool_call_id == ident:
                return value.model_copy(update={'additional_kwargs': {
                    **value.additional_kwargs, ANNOTATION: fields}})
            return value
        if isinstance(result, ToolMessage):
            return message(result)
        if isinstance(result, Command) and result.graph is None and isinstance(result.update, dict):
            messages = result.update.get('messages')
            if isinstance(messages, ToolMessage):
                messages = message(messages)
            elif isinstance(messages, list):
                messages = [message(m) for m in messages]
            else:
                return result
            return replace(result, update={**result.update, 'messages': messages})
        return result

    def wrap_tool_call(self, request, handler):
        normalized, fields = self._request(request)
        return self._annotate(handler(normalized), fields, request.tool_call['id'])

    async def awrap_tool_call(self, request, handler):
        normalized, fields = self._request(request)
        return self._annotate(await handler(normalized), fields, request.tool_call['id'])
