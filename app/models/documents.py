from sqlmodel import Field, SQLModel


class Documents(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)
    doc_id: str = Field(unique=True)
    name: str
    file_path: str
    status: str = Field(default="pending")
    description: str | None = None
