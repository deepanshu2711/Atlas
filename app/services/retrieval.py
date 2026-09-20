from typing import Literal

from langchain_core.documents import Document

from app.core.config import settings
from app.services.bm25 import bm25_search
from app.services.filters import doc_filter
from app.utils.store import vector_store, vector_store_v2

RetrievalMode = Literal["dense", "bm25", "hybrid"]


def dense_search(query: str, doc_id: str, k: int, use_v2: bool) -> list[Document]:
    store = vector_store_v2 if use_v2 else vector_store
    return store.similarity_search(query=query, k=k, filter=doc_filter(doc_id))


def _chunk_key(doc: Document) -> tuple[str, int]:
    return doc.metadata["doc_id"], doc.metadata["chunk_index"]


def rrf_fuse(rankings: list[list[Document]], rrf_k: int) -> list[Document]:
    """Reciprocal Rank Fusion: each chunk scores sum(1 / (rrf_k + rank)) over
    the rankings it appears in (rank starts at 1). Only ranks matter, so the
    incomparable dense and BM25 scores never need to be normalised."""
    scores: dict[tuple[str, int], float] = {}
    docs: dict[tuple[str, int], Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            key = _chunk_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs.setdefault(key, doc)
    # sorted() is stable, so equal scores keep first-seen order (dense first)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def retrieve(
    query: str,
    doc_id: str,
    use_v2: bool,
    k: int | None = None,
    mode: RetrievalMode | None = None,
) -> list[Document]:
    """Return up to `k` chunks of one document, best first.

    `k` and `mode` default to settings; callers (e.g. the eval harness) pass
    them explicitly to compare configurations without touching the environment.
    """
    k = settings.final_k if k is None else k
    mode = mode or settings.retrieval_mode

    if mode == "dense":
        return dense_search(query, doc_id, k, use_v2)
    if mode == "bm25":
        return bm25_search(query, doc_id, k, use_v2)
    if mode == "hybrid":
        n = settings.candidate_k
        fused = rrf_fuse([
            dense_search(query, doc_id, n, use_v2),
            bm25_search(query, doc_id, n, use_v2),
        ], settings.rrf_k)
        return fused[:k]
    raise NotImplementedError(f"retrieval_mode={mode!r} is not implemented yet")
