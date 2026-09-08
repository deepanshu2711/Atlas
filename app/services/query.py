from fastapi import HTTPException
from sqlmodel import Session

from app.repositories.documents import DocumentsRepository
from app.schemas.query import QueryPayload


class QueryService:
    def __init__(self, session: Session) -> None:
        self.document_repository = DocumentsRepository(session)

    async def query(self, payload: QueryPayload):
        document = self.document_repository.find_by_id(payload.document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
