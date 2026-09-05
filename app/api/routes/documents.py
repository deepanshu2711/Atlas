from fastapi import APIRouter

from app.model.schema.documents import DocumentAdd
from app.services.documents import DocumentsService
from app.utils.database import SessionDep

router = APIRouter()


@router.post("/")
def submit_document(payload: DocumentAdd, session: SessionDep):
    service = DocumentsService(session)
    return service.create(payload)


@router.get("/")
def get_all(session: SessionDep):
    service = DocumentsService(session)
    return service.all()
