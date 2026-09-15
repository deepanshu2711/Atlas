from fastapi import APIRouter

from app.core.database import SessionDep
from app.schemas.query import QueryPayload
from app.services.query import QueryService

router = APIRouter(
    tags=["query"],
)


@router.post("")
async def query(payload: QueryPayload, session: SessionDep):
    service = QueryService(session)
    return await service.query(payload)
