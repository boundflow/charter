from types import SimpleNamespace
import pytest
from charter.harness.context import with_context_profile


def test_local_context_reaches_summarization_and_reserves_output():
    from deepagents.middleware.summarization import compute_summarization_defaults
    wrapper = SimpleNamespace(chat_model=SimpleNamespace(num_ctx=32768, profile=None),
                              governor=SimpleNamespace(max_tokens=8192), profile=None)
    assert with_context_profile(wrapper) is wrapper
    assert wrapper.profile['max_input_tokens'] == 24576
    assert compute_summarization_defaults(wrapper)['trigger'] == ('fraction', .85)
    # The same model object is inherited by subagents and remains governed.
    assert wrapper.governor.max_tokens == 8192


def test_provider_profile_is_preserved_and_never_expanded():
    profile = {'max_input_tokens': 12000, 'tool_calling': True}
    wrapper = SimpleNamespace(chat_model=SimpleNamespace(num_ctx=32768, profile=profile),
                              governor=SimpleNamespace(max_tokens=8192), profile=None)
    with_context_profile(wrapper)
    assert wrapper.profile['max_input_tokens'] == 12000
    assert wrapper.profile['tool_calling'] is True
    assert 'max_output_tokens' not in profile


def test_impossible_context_fails_explicitly():
    wrapper = SimpleNamespace(chat_model=SimpleNamespace(num_ctx=8192, profile=None),
                              governor=SimpleNamespace(max_tokens=8192), profile=None)
    with pytest.raises(ValueError, match='must exceed output allowance'):
        with_context_profile(wrapper)


def test_unknown_models_are_not_assigned_invented_limits():
    model = SimpleNamespace(profile=None)
    assert with_context_profile(model).profile is None


def test_real_governance_wrapper_accepts_the_effective_profile():
    ollama = pytest.importorskip('langchain_ollama')
    from boundflow.langchain_client import GovernedChatModel
    model = GovernedChatModel(governor=SimpleNamespace(max_tokens=8192),
                              chat_model=ollama.ChatOllama(model='qwen3.5:9b', num_ctx=32768))
    assert with_context_profile(model).profile['max_input_tokens'] == 24576


def test_reminder_reservation_reduces_history_space_without_accumulating():
    inner = SimpleNamespace(num_ctx=32768, profile={'max_input_tokens': 12000, 'tool_calling': True})
    wrapper = SimpleNamespace(chat_model=inner, governor=SimpleNamespace(max_tokens=4096), profile=None)
    for _ in range(2):
        with_context_profile(wrapper, reserved_input_tokens=1800)
        assert wrapper.profile['max_input_tokens'] == 10200
        assert wrapper.profile['max_output_tokens'] == 4096
    assert inner.profile['max_input_tokens'] == 12000
    with pytest.raises(ValueError, match='reserved caller context'):
        with_context_profile(wrapper, reserved_input_tokens=11500)


@pytest.mark.parametrize('invalid', [-1, True, 2.5])
def test_invalid_reminder_reservation_is_rejected(invalid):
    with pytest.raises(ValueError, match='nonnegative integer'):
        with_context_profile(SimpleNamespace(profile=None), reserved_input_tokens=invalid)
