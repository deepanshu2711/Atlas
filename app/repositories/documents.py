from sqlmodel import Session, select

from app.utils.database import Documents


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
