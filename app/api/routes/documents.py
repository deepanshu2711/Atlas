from fastapi import APIRouter, BackgroundTasks, File, UploadFile, status

from app.core.database import SessionDep
from app.services.documents import DocumentsService

router = APIRouter(
    tags=["Documents"],
)


@router.post("", status_code=status.HTTP_201_CREATED)
async def submit_document(
    session: SessionDep,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    service = DocumentsService(session)
    return await service.create(file=file, background_tasks=background_tasks)


@router.get("")
def get_all(session: SessionDep):
    service = DocumentsService(session)
    return service.all()
