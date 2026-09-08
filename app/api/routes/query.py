from fastapi import APIRouter

from app.model.schema.query import QueryPayload
from app.utils.database import SessionDep

router = APIRouter()


@router.post("")
async def query(payload: QueryPayload, session: SessionDep):
    return
