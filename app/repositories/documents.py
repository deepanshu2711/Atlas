from sqlmodel import Session, select

from app.models.documents import Documents


class DocumentsRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, payload: Documents) -> Documents:
        self.session.add(payload)
        self.session.commit()
        self.session.refresh(payload)
        return payload

    def all(self):
        return self.session.exec(select(Documents)).all()

    def find_by_id(self, id: int):
        statement = select(Documents).where(Documents.id == id)
        return self.session.exec(statement=statement).first()

    def update(self, document: Documents) -> Documents:
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document
