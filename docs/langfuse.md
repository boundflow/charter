# Native Langfuse tracing

Install `pip install -e '.[langfuse]'` and add to `worker.yaml`:

```yaml
langfuse:
  session_input: project_id
```

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` in the
worker environment. Use the base URL matching your Langfuse Cloud project region.
`session_input` optionally names a task input used to group runs into a session;
when absent or missing on a task, the workflow instance ID is the session ID.

The worker adds a fresh native Langfuse LangChain callback for each invocation,
alongside the existing governance callbacks. It traces the Deep Agents graph,
in-process subagents, model messages and usage, and tool inputs/results. The
worker supplies the configured model name as callback metadata. Specialist model
overrides and durable child workflows need separate integration verification.
The governed model wrapper is represented as a chain step, so only the actual
model call contributes generation usage/cost. Tool wrappers may appear alongside
their underlying tool calls in the execution tree.

Each task has a stable trace ID derived from its original task ID. Resumed rounds
emit separate real roots within that trace. Independent runs in the same project
share a session, not a trace. Agent name/version, task ID and workflow ID are
attached as metadata. This is execution history, not persistent agent memory.

Langfuse uses its own tracer provider and HTTP exporter; a Collector is not needed.
Avoid also sending `trace_sink: {kind: otel}` to the same Langfuse project: those
BoundFlow operation summaries overlap with the native model/tool observations.
A separate JSONL sink can still be used for local operation records.

Sampling is set to 100%. The SDK batches delivery and flushes exports
when the worker exits normally. This is best-effort telemetry, not a guaranteed
audit journal; process crashes or network failures may lose spans. Approval
decisions remain in BoundFlow's audit log. Configure retention in Langfuse to
match your history needs. Captured messages and tool results go to that project.

Offline integration coverage: `pytest tests/test_langfuse.py`. It uses the real
Deep Agents graph, BoundFlow model wrapper, Langfuse callback and in-memory OTel
exporter. Two helpers run concurrently, one tool returns a handled failure, and
the tests check span ancestry, usage, session attribution and resumed task IDs.
Cloud ingestion and UI rendering still require a live test with your credentials.
