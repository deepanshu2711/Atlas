from typing import Literal

from langchain_core.documents import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings
from app.utils.store import vector_store, vector_store_v2

RetrievalMode = Literal["dense", "bm25", "hybrid"]


def _doc_filter(doc_id: str) -> Filter:
    return Filter(must=[
        FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))
    ])


def dense_search(query: str, doc_id: str, k: int, use_v2: bool) -> list[Document]:
    store = vector_store_v2 if use_v2 else vector_store
    return store.similarity_search(query=query, k=k, filter=_doc_filter(doc_id))


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
    raise NotImplementedError(f"retrieval_mode={mode!r} is not implemented yet")
