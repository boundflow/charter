"""Narrow compatibility parser for Qwen-coder's missing opening tool_call tag.

Only a whole assistant message containing one fully closed native function fragment
is eligible. Prose, examples, mixed native calls, unknown tools, duplicate parameters,
and invalid arguments are rejected. This returns data to the ordinary governed tool
loop; it never invokes a tool or changes its authorization.
"""
import hashlib
import json
import re
from uuid import uuid4

from jsonschema import Draft202012Validator
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Non-JSON numeric constant')


def normalize(message, tools):
    if (not isinstance(message, AIMessage) or message.tool_calls or message.invalid_tool_calls
            or not isinstance(message.content, str) or len(message.content) > 1_000_000):
        return None
    fragment = re.fullmatch(r'\s*<function=([\w:-]+)>\s*([\s\S]*?)\s*</function>\s*</tool_call>\s*', message.content)
    if not fragment:
        return None
    name, body = fragment.groups()
    schemas = [convert_to_openai_tool(tool)['function'] for tool in tools]
    matches = [schema for schema in schemas if schema['name'] == name]
    if len(matches) != 1:
        return None
    schema = matches[0].get('parameters', {'type': 'object', 'properties': {}})
    properties = schema.get('properties', {})
    arguments = {}
    while body.strip():
        parameter = re.match(r'\s*<parameter=([\w-]+)>([\s\S]*?)</parameter>', body)
        if not parameter:
            return None
        key, value = parameter.groups()
        if key not in properties or key in arguments:
            return None
        # Native delimiter tokens inside a value are ambiguous; require a new model call.
        if re.search(r'</?(?:tool_call|function|parameter)(?:[=>]|\s)', value):
            return None
        value = value.removeprefix('\r\n').removeprefix('\n')
        value = value.removesuffix('\r\n').removesuffix('\n')
        field = properties[key]
        types = set()
        for node in [field, *field.get('anyOf', []), *field.get('oneOf', [])]:
            value_type = node.get('type', [])
            types.update([value_type] if isinstance(value_type, str) else value_type)
        try:
            arguments[key] = value if 'string' in types else json.loads(
                value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (ValueError, TypeError, RecursionError):
            return None
        body = body[parameter.end():]
    if not Draft202012Validator({**schema, 'additionalProperties': False}).is_valid(arguments):
        return None
    return message.model_copy(update={
        'content': '',
        'tool_calls': [{'name': name, 'args': arguments, 'id': 'qwen_' + uuid4().hex, 'type': 'tool_call'}],
        'additional_kwargs': {**message.additional_kwargs, 'tool_format_normalization': {
            'parser': 'qwen_missing_opener_v1',
            'raw_sha256': hashlib.sha256(message.content.encode()).hexdigest(),
            'tool': name}},
    })
