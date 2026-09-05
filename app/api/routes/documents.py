from fastapi import APIRouter, File, UploadFile

from app.services.documents import DocumentsService
from app.utils.database import SessionDep

router = APIRouter()


@router.post("/")
async def submit_document(session: SessionDep, file: UploadFile = File(...)):
    service = DocumentsService(session)
    return await service.create(file=file)


@router.get("/")
def get_all(session: SessionDep):
    service = DocumentsService(session)
    return service.all()
