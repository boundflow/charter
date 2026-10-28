"""Bounded retry plus optional strict Qwen native-fragment normalization."""
import logging
import re
from dataclasses import replace

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger(__name__)


class MalformedToolOutput(RuntimeError):
    pass


class ToolFormatRecovery(AgentMiddleware):
    def __init__(self, retries, normalize_qwen=False):
        self.retries = retries
        self.normalize_qwen = normalize_qwen

    def _normalize(self, request, response):
        if self.normalize_qwen and len(response.result) == 1:
            from .qwen_tool_fragment import normalize
            normalized = normalize(response.result[0], request.tools)
            if normalized is not None:
                log.warning('Normalized a complete Qwen native fragment for %s; ordinary tool governance still applies',
                            normalized.tool_calls[0]['name'])
                return replace(response, result=[normalized])
        return response

    def _retry(self, request, response, attempt):
        names = set()
        for tool in request.tools:
            if isinstance(tool, dict):
                names.add(tool.get('name') or tool.get('function', {}).get('name'))
            else:
                names.add(tool.name)
        message = next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)
        if message is None or message.tool_calls or not isinstance(message.content, str):
            return None
        claimed = set(re.findall(r'<function=([\w:-]+)>', message.content)) & names
        if not claimed:
            return None
        if attempt >= self.retries:
            raise MalformedToolOutput('Model repeatedly printed tool markup instead of a structured tool call; no printed action was executed: ' + ', '.join(sorted(claimed)))
        log.warning('Recovering malformed tool output for %s (retry %d/%d); printed action was not executed',
                    ', '.join(sorted(claimed)), attempt + 1, self.retries)
        feedback = HumanMessage(content=(
            'Your last response printed malformed tool-call markup as ordinary text. No tool ran. '
            'Retry the advertised native tool invocation directly without a prose preamble. '
            'Preserve the complete provider function-call wrapper, including the opening '
            '<tool_call> before <function=...>; do not omit its opening tag. '
            'Preserve the original request and tool restrictions. Do not claim the action happened.'))
        return request.override(messages=[*request.messages, message, feedback])

    def wrap_model_call(self, request, handler):
        for attempt in range(self.retries + 1):
            response = self._normalize(request, handler(request))
            retry = self._retry(request, response, attempt)
            if retry is None:
                return response
            request = retry

    async def awrap_model_call(self, request, handler):
        for attempt in range(self.retries + 1):
            response = self._normalize(request, await handler(request))
            retry = self._retry(request, response, attempt)
            if retry is None:
                return response
            request = retry
