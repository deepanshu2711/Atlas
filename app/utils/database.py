from sqlmodel import Field, SQLModel, create_engine


class Documents(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)
    name: str
    status: str = Field(default="pending")
    description: str | None = None


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
