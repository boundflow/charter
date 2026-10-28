"""Recover consecutive, unchanged reads without hiding or inventing tool results."""
import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from .delegation_prerequisites import _successful


class RepeatedReadLoop(RuntimeError):
    pass


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(b, dict) and b.get('type') == 'text' for b in content):
        # Transport-generated block IDs are not changes to the returned evidence.
        return '\n'.join(b.get('text', '') for b in content)
    return None


class ReadLoopRecovery(AgentMiddleware):
    def __init__(self, read_tools):
        self.read_tools = frozenset(read_tools)

    def _request(self, request):
        messages = request.messages
        count, previous, name = 0, None, None
        index = len(messages) - 1
        while index >= 1:
            results = {}
            invalid = False
            while index >= 0 and isinstance(messages[index], ToolMessage):
                result = messages[index]
                if result.tool_call_id in results:
                    invalid = True
                    break
                results[result.tool_call_id] = result
                index -= 1
            if invalid or index < 0:
                break
            call_message = messages[index]
            if not isinstance(call_message, AIMessage) or not call_message.tool_calls:
                break
            calls = call_message.tool_calls
            if len(calls) != len(results) or {c['id'] for c in calls} != set(results):
                break
            signatures = []
            for call in calls:
                text = _text(results[call['id']].content)
                if call['name'] not in self.read_tools or text is None:
                    break
                signatures.append((call['name'], json.dumps(call['args'], sort_keys=True),
                                   _successful(results[call['id']]), text))
            if len(signatures) != len(calls):
                break
            # Match the actual argument/result pairs even when parallel results
            # or the next batch's calls arrive in a different order.
            signature = tuple(sorted(signatures))
            if previous is not None and signature != previous:
                break
            previous = signature
            name = ', '.join(sorted({c['name'] for c in calls}))
            count += 1
            index -= 1
        unit = 'reads' if previous and len(previous) == 1 else 'read batches'
        outcome = 'successful' if previous and all(item[2] for item in previous) else 'failed'
        if count >= 6:
            raise RepeatedReadLoop(
                f'No progress: {count} consecutive identical {outcome} {unit} from {name}, '
                'despite recovery feedback. Results are retained; no completion is claimed.')
        if count < 3:
            return request
        feedback = HumanMessage(content=(
            f'Execution feedback: the last {count} completed {unit} used identical arguments '
            f'to {name} and returned identical {outcome} content. No new evidence was obtained. '
            'The most recent result is already in this conversation. Continue from that result: '
            'choose a different useful action, such as reading a missing input or delegating when '
            'prerequisites are satisfied. If you believe repeating is necessary, explain what changed. '
            'All existing requirements and tool restrictions still apply.'))
        return request.override(messages=[*messages, feedback])

    def wrap_model_call(self, request, handler):
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request(request))
