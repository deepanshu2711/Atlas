from langchain_core.documents import Document

from app.core.logging import get_logger
from app.services.bm25 import chunks_for
from app.services.graph_extract import build_extract_llm, extract_chunk_graph
from app.utils.graph_store import get_graph_store

logger = get_logger(__name__)


def index_document(doc_id: str, chunks: list[tuple[int, str]]) -> int:
    """Extract entities/relations from each (chunk_index, text) with the LLM
    and replace the document's graph in Kuzu. Returns the entity count."""
    llm = build_extract_llm()
    extracted = [extract_chunk_graph(llm, idx, text) for idx, text in chunks]
    return get_graph_store().replace_document(doc_id, extracted)


def graph_search(query: str, doc_id: str, k: int, use_v2: bool) -> list[Document]:
    """Chunks linked to the entities named in `query`, best first.

    The graph is built from v2 chunks only (their chunk_index is what the
    Chunk nodes store), so v1 lookups return nothing. A question that names
    no known entity also returns nothing, which RRF treats as "no opinion"."""
    if not use_v2:
        return []
    store = get_graph_store()
    entity_ids = store.match_entities(doc_id, query)
    scores = store.chunk_scores(entity_ids)
    if not scores:
        return []
    by_index = {d.metadata.get("chunk_index"): d for d in chunks_for(doc_id, use_v2)}
    # highest score first; ties go to the earlier chunk, like bm25 does
    ranked = sorted(scores, key=lambda idx: (-scores[idx], idx))
    return [by_index[idx] for idx in ranked if idx in by_index][:k]
