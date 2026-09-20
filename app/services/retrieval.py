from typing import Literal

from langchain_core.documents import Document

from app.core.config import settings
from app.services.bm25 import bm25_search
from app.services.filters import doc_filter
from app.utils.reranker import get_reranker
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


def rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    """Reorder `docs` by cross-encoder relevance to `query`; keep the top `k`."""
    if not docs:
        return []
    scores = get_reranker().predict([(query, d.page_content) for d in docs])
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in order[:k]]


def _first_stage(query: str, doc_id: str, use_v2: bool, k: int, mode: RetrievalMode) -> list[Document]:
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
    raise NotImplementedError(f"retrieval_mode={mode!r} is not implemented")


def retrieve(
    query: str,
    doc_id: str,
    use_v2: bool,
    k: int | None = None,
    mode: RetrievalMode | None = None,
    use_rerank: bool | None = None,
) -> list[Document]:
    """Return up to `k` chunks of one document, best first.

    `k`, `mode` and `use_rerank` default to settings; callers (e.g. the eval
    harness) pass them explicitly to compare configurations without touching
    the environment. With reranking, the first stage supplies `candidate_k`
    candidates (at least `k`) and the cross-encoder picks the top `k`.
    """
    k = settings.final_k if k is None else k
    mode = mode or settings.retrieval_mode
    use_rerank = settings.rerank_enabled if use_rerank is None else use_rerank

    if not use_rerank:
        return _first_stage(query, doc_id, use_v2, k, mode)
    candidates = _first_stage(query, doc_id, use_v2, max(k, settings.candidate_k), mode)
    return rerank(query, candidates, k)
