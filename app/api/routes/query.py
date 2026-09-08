from fastapi import APIRouter

from app.core.database import SessionDep
from app.schemas.query import QueryPayload

router = APIRouter()


@router.post("")
async def query(payload: QueryPayload, session: SessionDep):
    return
