import re
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.utils.qdrant import COLLECTION_NAME, COLLECTION_NAME_v2, client
from app.services.filters import doc_filter

_TOKEN_RE = re.compile(r"\w+")
_SCROLL_BATCH = 256


@dataclass
class _Index:
    bm25: BM25Okapi
    docs: list[Document]


# One index per (collection, doc_id), built lazily from the chunks already in
# Qdrant so it indexes exactly the text that gets embedded.
_indexes: dict[tuple[str, str], _Index] = {}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_chunks(collection: str, doc_id: str) -> list[Document]:
    docs, offset = [], None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=doc_filter(doc_id),
            limit=_SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        docs.extend(
            Document(page_content=p.payload["page_content"],
                     metadata=p.payload.get("metadata", {}))
            for p in points
        )
        if offset is None:
            break
    # scroll order is by point id (random uuid5); sort for a stable tie-break
    docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    return docs


def _get_index(collection: str, doc_id: str) -> _Index | None:
    key = (collection, doc_id)
    if key not in _indexes:
        docs = _load_chunks(collection, doc_id)
        if not docs:
            return None
        _indexes[key] = _Index(
            bm25=BM25Okapi([tokenize(d.page_content) for d in docs]), docs=docs)
    return _indexes[key]


def invalidate(doc_id: str) -> None:
    """Drop cached indexes for a document; call after (re-)ingesting it."""
    for collection in (COLLECTION_NAME, COLLECTION_NAME_v2):
        _indexes.pop((collection, doc_id), None)


def bm25_search(query: str, doc_id: str, k: int, use_v2: bool) -> list[Document]:
    index = _get_index(COLLECTION_NAME_v2 if use_v2 else COLLECTION_NAME, doc_id)
    if index is None:
        return []
    scores = index.bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    # a chunk sharing no query term scores 0; returning it would be noise
    return [index.docs[i] for i in ranked[:k] if scores[i] > 0]
