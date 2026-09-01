"""The trace sink `worker.yaml` declares.

A trace is the run tree — every model call and tool call, with prompts, results and
token counts. It is captured in the worker and shipped to a sink you own, so it
lands in your backend and never reaches the control plane.

Separate from the store: `store.url` holds checkpoints and the agent's files, which
is state a parked task resumes from, not telemetry.
"""

from __future__ import annotations

from .config.worker import TraceSink


def build(spec: TraceSink | None, service: str = "charter"):
    """The sink for a manifest, or None when it declares none."""
    if spec is None or spec.kind == "none":
        return None

    from boundflow.trace import (JsonlFileTraceSink, LoggingTraceSink,
                                 OTelTraceSink)

    from .worker import resolve

    if spec.kind == "logging":
        return LoggingTraceSink()
    if spec.kind == "jsonl":
        return JsonlFileTraceSink(resolve(spec.path))
    return OTelTraceSink(_otlp_tracer(resolve(spec.endpoint), service))


def _otlp_tracer(endpoint: str, service: str):
    """An OTLP exporter on `endpoint`, registered as the global tracer provider.

    `http://` sends without TLS; anything else uses it. The provider is global
    because a harness or library emitting its own spans should land in the same
    trace as the run that caused them.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter)
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        raise ImportError(
            "trace_sink kind `otel` needs its optional dependencies:\n"
            "    pip install 'charter[otel]'"
        ) from e

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"))))
    trace.set_tracer_provider(provider)
    return provider.get_tracer("charter")
