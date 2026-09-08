from pydantic import BaseModel


class DocumentAdd(BaseModel):
    name: str
    description: str | None
