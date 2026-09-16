# Local models

Inference is bring-your-own, so a model on your own machine is a provider like any
other. The control plane governs the run either way — what it records is which
tools were called, what was approved, whether it succeeded. Prompts and model
traffic never reach it, from any provider.

## Configuration

```bash
pip install 'boundflow-charter[ollama]'
ollama pull qwen2.5:7b
```

In `worker.yaml`:

```yaml
llm:
  provider: ollama        # no api_key: a local runtime needs none
```

The model name is in the agent's version file, like any other model:

```yaml
model: qwen2.5:7b
```

Needs `boundflow>=0.7.4`. Below that, every call fails with `TypeError:
AsyncClient.chat() got an unexpected keyword argument 'max_tokens'`.

`provider` takes anything LangChain can build, so an OpenAI-compatible server you
host is `provider: openai` with `base_url` pointing at it. Only Ollama has been
run end to end.

## What differs from a hosted provider

Measured running the quickstart `triage` agent against Ollama 0.34.0 on CPU.

**Calls need more time.** `max_call_seconds` defaults to 60. A single quickstart
call took about 170 seconds. Raise it in `runtime.yaml`:

```yaml
limits:
  max_call_seconds: 600
```

**Budget in calls, not dollars.** An unpriced model reports no cost, so
`max_cost_usd` never binds and a run has no ceiling until `max_llm_calls` does:

```yaml
per_run:
  max_llm_calls: 12
```

Usage is still reported — Ollama returns `usage_metadata` on plain, tool-bound and
streamed calls — so runs are metered and appear in the console. The cost is $0.

**Size the model for tool calling.** The agent drives itself through tools, and
ending a run means calling `submit_result`.

| Model | Result |
|---|---|
| `qwen2.5:7b` | completes the quickstart, 46s to 3m depending on the machine |
| `qwen2.5:3b` | completes calls, doesn't follow the harness — it answered the ticket by writing a file instead of calling `submit_result`, with prompts well inside its context window |

A model that never calls `submit_result` burns its call budget and the run is
recorded as a failure, which is what the lifecycle rules act on.

Forcing the ending is weaker here than elsewhere: on the last permitted call
Charter offers only `submit_result`, but Ollama cannot force a tool at all, so a
model can still reply in text instead.

## Context window

The window belongs to the server, not to Charter, and Ollama's default is small:

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

4,096 was enough for the quickstart. An agent with tools and a longer conversation
wants more.

## Testing

`tests/e2e/test_ollama.py` runs the quickstart scaffold against a local model with
nothing faked, including the provider Charter builds from `worker.yaml`:

```bash
OLLAMA_TEST_MODEL=qwen2.5:7b pytest tests/e2e/test_ollama.py
```

It skips without that variable, and is not in CI: a model small enough to pull on
every run is too small to drive the harness.
