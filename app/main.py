from fastapi import FastAPI
from app.core.logging import configure_logging
from app.api.routes.welcome import router as welcome_router
from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router

configure_logging()

app = FastAPI()

app.include_router(welcome_router)
app.include_router(documents_router, prefix="/api/v1/documents")
app.include_router(query_router, prefix="/api/v1/query")
