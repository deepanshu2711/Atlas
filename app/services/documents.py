import uuid
from pathlib import Path
from pypdf import PdfReader
from fastapi import HTTPException, UploadFile
from sqlmodel import Session
from app.models.documents import Documents
from app.repositories.documents import DocumentsRepository
from app.utils.store import vector_store


RAW_DIR = Path("data/raw_pdfs")
ALLOWED_CONTENT_TYPE = "application/pdf"
CHUNK_SIZE = 1024 * 1024


class DocumentsService:
    def __init__(self, session: Session):
        self.repository = DocumentsRepository(session)
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def all(self):
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
            document = self.repository.create(document)
        except Exception:
            if dest.exists():
                dest.unlink()
            raise

        try:
            self.ingest_document(doc_id=doc_id, file_path=str(dest))
            document.status = "ready"
        except Exception:
            document.status = "failed"
        finally:
            document = self.repository.update(document)

        return document

    # NOTE: Methods that does not require self
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

    @staticmethod
    def ingest_document(doc_id: str, file_path: str, chunk_size: int = 512) -> int:
        reader = PdfReader(file_path)
        full_text = "\n".join(page.extract_text()
                              or "" for page in reader.pages)
        chunks = [full_text[i:i+chunk_size]
                  for i in range(0, len(full_text), chunk_size)]

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}"))
            for idx in range(len(chunks))
        ]
        metadatas = [
            {"doc_id": doc_id, "chunk_index": idx}
            for idx in range(len(chunks))
        ]
        vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

        return len(chunks)
