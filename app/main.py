from fastapi import FastAPI
from api.routes.welcome import router as welcome_router
from api.routes.documents import router as documents_router

app = FastAPI()

app.include_router(welcome_router)
app.include_router(documents_router, prefix="/api/v1/documents")
