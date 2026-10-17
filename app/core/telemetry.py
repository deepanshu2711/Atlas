import threading
import json
from opentelemetry.trace import format_trace_id, format_span_id
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor, SpanExporter, SpanExportResult)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry import trace
from pathlib import Path

from app.core.config import settings


class JsonlSpanExporter(SpanExporter):
    """One file per trace: SPAN_DIR/<trace_id>.jsonl"""

    def __init__(self, directory: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans):
        with self._lock:
            for s in spans:
                ctx = s.get_span_context()
                rec = {
                    "trace_id": format_trace_id(ctx.trace_id),
                    "span_id": format_span_id(ctx.span_id),
                    "parent_id": format_span_id(s.parent.span_id) if s.parent else None,
                    "name": s.name,
                    "service": s.resource.attributes.get("service.name"),
                    "start_ns": s.start_time,
                    "end_ns": s.end_time,
                    "status": s.status.status_code.name,      # OK | ERROR | UNSET
                    "attributes": dict(s.attributes or {}),
                }
                with open(self.dir / f"{rec['trace_id']}.jsonl", "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def setup_tracing(service_name: str):
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(
        SimpleSpanProcessor(JsonlSpanExporter(settings.span_dir)))
    trace.set_tracer_provider(provider)
