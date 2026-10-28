"""Expose an optional caller deadline without changing any enforcement limit."""
from datetime import datetime, timezone
import math
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage


class CallerDeadline(AgentMiddleware):
    def __init__(self, deadline, now=None):
        if not isinstance(deadline, str):
            raise ValueError('Caller deadline must be an ISO timestamp with timezone')
        try:
            parsed = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Caller deadline must be an ISO timestamp with timezone') from None
        if parsed.tzinfo is None:
            raise ValueError('Caller deadline must include a timezone')
        self.deadline = parsed.astimezone(timezone.utc)
        self.now = now or (lambda: datetime.now(timezone.utc))

    def message(self):
        remaining = max(0, math.ceil((self.deadline - self.now()).total_seconds()))
        return HumanMessage(content=(
            f'CALLER DEADLINE (runtime metadata, not new instructions): {self.deadline.isoformat()}; '
            f'{remaining} seconds remain in the caller operation. This may be shorter than the agent policy ceiling. '
            'Complete a coherent bounded change and allow time for required verification and handoff. '
            'If a needed integration path is outside ownership, return its exact path and blocker promptly. '
            'Continue from actual receipts; avoid restarting completed inspection. Do not substitute example-specific '
            'behavior for the requested integration. Report incomplete work honestly when time is insufficient. '
            'This notice grants no permissions, changes no enforcement limit, and never waives fresh review or acceptance.'))

    def reserved_tokens(self):
        from langchain_core.messages.utils import count_tokens_approximately
        return count_tokens_approximately([self.message()]) + 32

    def _request(self, request):
        return request.override(messages=[*request.messages, self.message()])

    def wrap_model_call(self, request, handler):
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request(request))
