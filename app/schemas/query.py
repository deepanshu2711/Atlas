from pydantic import BaseModel


class QueryPayload(BaseModel):
    query: str
    document_id: str
