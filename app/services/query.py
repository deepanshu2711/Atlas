from fastapi import HTTPException
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlmodel import Session

from app.repositories.documents import DocumentsRepository
from app.schemas.query import QueryPayload
from app.utils.store import vector_store


class QueryService:
    def __init__(self, session: Session) -> None:
        self.document_repository = DocumentsRepository(session)

    async def query(self, payload: QueryPayload):
        document = self.document_repository.find_by_id(payload.document_id)
        print('document', document)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        # NOTE: retrive context
        docs = vector_store.similarity_search(query=payload.query, k=3, filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=payload.document_id)
                )
            ]
        ))

        print('retrive docs', docs)
        return
