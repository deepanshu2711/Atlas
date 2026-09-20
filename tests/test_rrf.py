from langchain_core.documents import Document

from app.services.retrieval import rrf_fuse


def _doc(idx: int) -> Document:
    return Document(page_content=f"chunk {idx}", metadata={"doc_id": "d", "chunk_index": idx})


def _ids(docs):
    return [d.metadata["chunk_index"] for d in docs]


def test_chunk_in_both_rankings_beats_chunk_in_one():
    dense = [_doc(1), _doc(2), _doc(3)]
    bm25 = [_doc(3), _doc(4)]
    # 3: 1/63 + 1/61 ; 1: 1/61 ; 4: 1/62 ; 2: 1/62
    assert _ids(rrf_fuse([dense, bm25], rrf_k=60))[0] == 3


def test_scores_match_formula():
    fused = rrf_fuse([[_doc(1), _doc(2)], [_doc(2), _doc(1)]], rrf_k=60)
    # both docs score 1/61 + 1/62; tie keeps first-seen order
    assert _ids(fused) == [1, 2]


def test_deduplicates_and_handles_empty_ranking():
    fused = rrf_fuse([[_doc(1), _doc(2)], []], rrf_k=60)
    assert _ids(fused) == [1, 2]
    assert rrf_fuse([[], []], rrf_k=60) == []
