# Fix slow / GPU+CPU-heavy document ingestion

## Context

The user reports that ingesting a ~100-page PDF takes a long time and drives up both GPU and CPU usage. An Explore pass through the ingestion pipeline (`app/services/documents.py`, `app/utils/embedding.py`, `app/utils/store.py`, `app/utils/qdrant.py`, `app/api/routes/documents.py`) plus direct reads of those files found three concrete, fixable causes — all in the active `ingest_document_v2` path used by `DocumentsService.create`. The goal is to cut ingestion time/resource usage and stop the API from appearing to hang while a document is processing.

**Not the cause:** embedding generation (`bge-small-en-v1.5` via `sentence-transformers`) and the Qdrant upsert are already properly batched (batch size 64, plus `encode()`'s own internal batching) — no change needed there.

## Root causes (ranked by impact)

1. **Docling runs full OCR + table-structure detection on every page, unconditionally.**
   `DocumentConverter().convert(file_path)` (`app/services/documents.py:116`) uses Docling's default `PdfPipelineOptions`, which has `do_ocr=True` and `do_table_structure=True`. That means every page of every upload — even a plain text-layer PDF with no tables — runs a full EasyOCR pass and a TableFormer table-structure pass, on top of layout analysis. These are the heaviest GPU/CPU deep-learning passes in the pipeline and are almost certainly the dominant cost for 100 pages.

2. **The Docling models are reloaded from disk (and re-placed on GPU) on every single upload.**
   `DocumentConverter()` and `HybridChunker()` are constructed fresh inside `ingest_document_v2` (`app/services/documents.py:116` and `:130`) instead of once at startup. `DocumentConverter.initialized_pipelines` is an instance-level cache, so a new instance means Docling's layout/table/OCR models are reinitialized from scratch every time — unlike `embedding_model` (`app/utils/embedding.py`) and `vector_store`/`vector_store_v2` (`app/utils/store.py`), which are already correctly built once as module-level singletons.

3. **Ingestion runs synchronously on the event loop, so the API looks fully hung during processing.**
   `submit_document` (`app/api/routes/documents.py:10`) and `DocumentsService.create` (`app/services/documents.py:31`) are `async def`, but the actual work — `self.ingest_document_v2(...)` at `documents.py:56` — is a plain blocking call, not offloaded to a thread. It blocks the single asyncio event loop for the whole ingestion duration, so no other request (health check, concurrent upload, or a query) can be served while a document is processing. This doesn't make ingestion itself faster, but it's very likely contributing to the "the whole thing is slow/stuck" perception.

Minor/related: `HybridChunker()`'s default tokenizer (`sentence-transformers/all-MiniLM-L6-v2`) is only used to enforce a token-length limit and doesn't match the actual embedding model (`BAAI/bge-small-en-v1.5`) — a config mismatch worth fixing while touching this code, not a perf issue on its own.

## Plan

### A. Make OCR / table-structure extraction configurable, off by default for text PDFs
- In `app/core/config.py`, add two settings (following the existing `Settings(BaseSettings)` pattern):
  ```python
  docling_ocr_enabled: bool = False
  docling_table_structure_enabled: bool = True
  ```
  (Table structure defaults to `True` since it's often wanted for real documents and cheaper than OCR; OCR defaults to `False` since the user's PDFs are normal text-based documents, not scans — confirmed with the user. Scanned PDFs can opt in via env var if that ever changes.)
- Build the converter with explicit `PdfPipelineOptions(do_ocr=settings.docling_ocr_enabled, do_table_structure=settings.docling_table_structure_enabled)` passed via `format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=...)}`.

### B. Build `DocumentConverter` and `HybridChunker` once, as module-level singletons
- Move their construction out of `ingest_document_v2` into module-level singletons in `app/services/documents.py` (mirroring `embedding_model` / `vector_store`), e.g.:
  ```python
  _pdf_pipeline_options = PdfPipelineOptions(
      do_ocr=settings.docling_ocr_enabled,
      do_table_structure=settings.docling_table_structure_enabled,
  )
  document_converter = DocumentConverter(
      format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_pipeline_options)}
  )
  chunker = HybridChunker(tokenizer="BAAI/bge-small-en-v1.5")
  ```
- Update `ingest_document_v2` to use these module-level instances instead of constructing new ones per call. This loads Docling's models once at process startup and reuses them (and the GPU placement) across uploads.
- Passing `tokenizer="BAAI/bge-small-en-v1.5"` also fixes the tokenizer mismatch noted above.

### C. Offload ingestion off the event loop
- In `DocumentsService.create` (`app/services/documents.py:56`), replace the direct call with a thread offload:
  ```python
  await asyncio.to_thread(self.ingest_document_v2, doc_id=doc_id, file_path=str(dest))
  ```
- Keeps the API responsive (health checks, concurrent queries/uploads) while a document processes. True async job/status polling (return immediately with `status="processing"`, client polls) is a larger change — flagged as an optional future follow-up, not part of this plan unless requested.

## Files to change
- `app/core/config.py` — add `docling_ocr_enabled`, `docling_table_structure_enabled` settings
- `app/services/documents.py` — module-level `document_converter`/`chunker` singletons with explicit `PdfPipelineOptions`, `asyncio.to_thread` wrap in `create()`, drop the now-redundant local construction in `ingest_document_v2`

## Verification
- `ingest_document_v2` already logs per-stage timings (`stage=convert`, `stage=chunk`, `stage=embed_upsert`, `stage=total`) — re-upload the same 100-page PDF before/after and compare `stage=convert elapsed=...` specifically; it should drop sharply once OCR/table-structure are disabled.
- Watch `nvidia-smi`/`htop` during an upload before and after to confirm reduced GPU/CPU load and shorter duration.
- While a document is uploading, hit `GET /api/v1/documents` (or the query endpoint) concurrently and confirm it no longer hangs until ingestion finishes.
- Re-run `evals/run_eval.py` to confirm retrieval quality is unaffected by the tokenizer alignment change.
