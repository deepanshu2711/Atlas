from fastapi import FastAPI
from api.routes.welcome import router as welcome_router


app = FastAPI()

app.include_router(welcome_router)
