from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def welcome():
    return {"message": "Welcome to Atlas API"}
