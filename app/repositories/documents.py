from sqlmodel import Session, select

from app.model.schema.documents import DocumentAdd
from app.utils.database import Documents


class DocumentsRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: DocumentAdd):
        document = Documents(**payload.model_dump())
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document

    def all(self):
        return self.session.exec(select(Documents)).all()
