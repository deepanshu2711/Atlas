from fastapi import APIRouter, Request
from opentelemetry import trace
from opentelemetry.propagate import extract

from app.core.database import SessionDep
from app.schemas.query import QueryPayload
from app.services.query import QueryService

router = APIRouter(
    tags=["query"],
)
tracer = trace.get_tracer("atlas")


@router.post("")
async def query(payload: QueryPayload, session: SessionDep, request: Request):
    with tracer.start_as_current_span("atlas.query", context=extract(request.headers), kind=trace.SpanKind.SERVER) as span:
        span.set_attribute("openinference.span.kind", "AGENT")
        span.set_attribute("input.value", payload.query)
        span.set_attribute("atlas.document_id", payload.document_id)
        result = await QueryService(session).query(payload=payload)
        span.set_attribute("output.value", result["answer"][:2000])
        span.set_attribute("atlas.mode", result["trace"]["mode"])

    return result
