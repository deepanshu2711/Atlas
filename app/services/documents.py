import asyncio
import logging
import time
import uuid
from pathlib import Path
from pypdf import PdfReader
from fastapi import HTTPException, UploadFile
from sqlmodel import Session
from app.core.config import settings
from app.core.logging import get_logger
from app.models.documents import Documents
from app.repositories.documents import DocumentsRepository
from app.utils.store import vector_store, vector_store_v2
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker


RAW_DIR = Path("data/raw_pdfs")
ALLOWED_CONTENT_TYPE = "application/pdf"
CHUNK_SIZE = 1024 * 1024
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

logger = get_logger(__name__)

# Built once at import time so Docling's layout/table/OCR models (and, on a
# GPU box, their CUDA placement) are loaded a single time and reused across
# every upload instead of being reinitialized from disk on each request.
_pdf_pipeline_options = PdfPipelineOptions(
    do_ocr=settings.docling_ocr_enabled,
    do_table_structure=settings.docling_table_structure_enabled,
)
document_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_pipeline_options),
    }
)
# Tokenizer matches the actual embedding model so the chunk-length limit
# reflects what will really be embedded (HybridChunker defaults to a
# different tokenizer otherwise).
chunker = HybridChunker(tokenizer=EMBEDDING_MODEL_NAME)


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
            save_start = time.perf_counter()
            await self._save_file(file, dest)
            logger.info(
                "document_create doc_id=%s stage=save_file elapsed=%.2fs",
                doc_id, time.perf_counter() - save_start,
            )
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
            # Docling conversion + embedding are CPU/GPU-bound blocking work;
            # run them in a worker thread so they don't stall the event loop
            # (and every other in-flight request) for the whole duration.
            await asyncio.to_thread(
                self.ingest_document_v2, doc_id=doc_id, file_path=str(dest))
            document.status = "ready"
        except Exception:
            logger.exception(
                "document_create doc_id=%s ingestion failed", doc_id)
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

    @staticmethod
    def ingest_document_v2(doc_id: str, file_path: str) -> int:
        total_start = time.perf_counter()

        convert_start = time.perf_counter()
        docling_logger = logging.getLogger("docling")
        prev_docling_level = docling_logger.level
        docling_logger.setLevel(logging.DEBUG)
        try:
            # docling emits PIPELINE_PROFILING debug logs naming the pages
            # in each processed batch, giving per-page conversion progress.
            doc = document_converter.convert(file_path).document
        finally:
            docling_logger.setLevel(prev_docling_level)
        convert_elapsed = time.perf_counter() - convert_start
        try:
            num_pages = len(doc.pages)
        except Exception:
            num_pages = None
        logger.info(
            "ingest_document_v2 doc_id=%s stage=convert elapsed=%.2fs pages=%s",
            doc_id, convert_elapsed, num_pages,
        )

        chunk_start = time.perf_counter()
        chunks = list(chunker.chunk(doc))
        logger.info(
            "ingest_document_v2 doc_id=%s stage=chunk elapsed=%.2fs chunks=%d",
            doc_id, time.perf_counter() - chunk_start, len(chunks),
        )

        texts, metadatas, ids = [], [], []

        for idx, chunk in enumerate(chunks):
            provenance = [
                {
                    "page_no": item.prov[0].page_no,
                    "bbox": {
                        "l": item.prov[0].bbox.l,
                        "t": item.prov[0].bbox.t,
                        "r": item.prov[0].bbox.r,
                        "b": item.prov[0].bbox.b
                    }
                }
                for item in chunk.meta.doc_items
                if item.prov
            ]
            texts.append(chunker.contextualize(chunk))
            metadatas.append({
                "doc_id": doc_id,
                "chunk_index": idx,
                "headings": chunk.meta.headings,
                "block_types": [item.label for item in chunk.meta.doc_items],
                "pages": sorted({p["page_no"] for p in provenance}),
                "provenance": provenance,
            })
            ids.append(
                str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:v2:{idx}")))

        upsert_start = time.perf_counter()
        vector_store_v2.add_texts(
            texts=texts, metadatas=metadatas, ids=ids)
        logger.info(
            "ingest_document_v2 doc_id=%s stage=embed_upsert elapsed=%.2fs chunks=%d",
            doc_id, time.perf_counter() - upsert_start, len(chunks),
        )

        logger.info(
            "ingest_document_v2 doc_id=%s stage=total elapsed=%.2fs pages=%s chunks=%d",
            doc_id, time.perf_counter() - total_start, num_pages, len(chunks),
        )

        return len(chunks)
