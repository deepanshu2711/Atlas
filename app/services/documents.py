from sqlmodel import Session

from app.model.schema.documents import DocumentAdd
from app.repositories.documents import DocumentsRepository


class DocumentsService:
    def __init__(self, session: Session):
        self.repository = DocumentsRepository(session)

    def all(self):
        return self.repository.all()

    def create(self, payload: DocumentAdd):
        return self.repository.create(payload)
