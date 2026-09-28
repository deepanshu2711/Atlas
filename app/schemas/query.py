from pydantic import BaseModel


class QueryPayload(BaseModel):
    query: str
    document_id: str
    use_v2: bool = True
    # None -> settings.agentic_enabled; the agentic loop needs v2 chunks
    agentic: bool | None = None
