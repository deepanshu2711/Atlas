import uuid
from pathlib import Path
from fastapi import HTTPException, UploadFile
from sqlmodel import Session
from app.repositories.documents import DocumentsRepository
from app.utils.database import Documents

RAW_DIR = Path("data/raw_pdfs")
ALLOWED_CONTENT_TYPE = "application/pdf"
CHUNK_SIZE = 1024 * 1024


class DocumentsService:
    def __init__(self, session: Session):
        self.repository = DocumentsRepository(session)
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Documents]:
        return self.repository.all()

    async def create(self, file: UploadFile) -> Documents:
        self._validate_file(file=file)

        doc_id = str(uuid.uuid4())
        dest = RAW_DIR / f"{doc_id}.pdf"

        try:
            await self._save_file(file, dest)
            document = Documents(
                doc_id=doc_id,
                file_path=str(dest),
                name=file.filename or ""
            )
            return self.repository.create(document)

        except Exception:
            if dest.exists():
                dest.unlink()
            raise

    @staticmethod
    def _validate_file(file: UploadFile) -> None:
        if file.content_type != ALLOWED_CONTENT_TYPE:
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted",
            )

    @staticmethod
    async def _save_file(
        file: UploadFile,
        destination: Path,
    ) -> None:
        with open(destination, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                buffer.write(chunk)
