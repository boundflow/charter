"""Carry the actual provider context budget through the governance wrapper."""
from __future__ import annotations


def with_context_profile(model, *, reserved_input_tokens=0):
    """Teach deepagents when to compact, while retaining the governed model.

    Local Ollama clients do not publish a profile. Without one, deepagents uses
    a 170K-token fallback even when num_ctx is 16K/32K. num_ctx covers both input
    and output, so reserve the governor's output cap before exposing input space.
    Caller reminders appended after compaction also reserve estimated input
    space; the model's history allowance must not consume their budget.
    The original messages remain in the durable harness's offloaded transcript.
    """
    if type(reserved_input_tokens) is not int or reserved_input_tokens < 0:
        raise ValueError('reserved_input_tokens must be a nonnegative integer')
    inner = getattr(model, 'chat_model', model)
    profile = dict(getattr(inner, 'profile', None) or {})
    context = getattr(inner, 'num_ctx', None)
    governor = getattr(model, 'governor', None)
    if isinstance(context, int) and not isinstance(context, bool) and context > 0:
        output = getattr(governor, 'max_tokens', None) or profile.get('max_output_tokens') or 4096
        existing = profile.get('max_input_tokens')
        available = context - output
        profile['max_input_tokens'] = min(existing, available) if isinstance(existing, int) else available
        profile['max_output_tokens'] = output
    if isinstance(profile.get('max_input_tokens'), int):
        profile['max_input_tokens'] -= reserved_input_tokens
        if profile['max_input_tokens'] < 1024:
            raise ValueError('Configured context must exceed output allowance and reserved caller context by at least 1024 tokens')
    if profile:
        model.profile = profile
    return model
