from langchain_core.documents import Document

from app.services import retrieval


class _FakeCrossEncoder:
    """Scores a pair by the digit in the chunk text, so order is predictable."""

    def predict(self, pairs):
        return [float(doc.split()[-1]) for _, doc in pairs]


def _docs(*scores: int) -> list[Document]:
    return [Document(page_content=f"chunk {s}", metadata={"doc_id": "d", "chunk_index": i})
            for i, s in enumerate(scores)]


def test_rerank_orders_by_score_and_cuts_to_k(monkeypatch):
    monkeypatch.setattr(retrieval, "get_reranker", lambda: _FakeCrossEncoder())
    out = retrieval.rerank("q", _docs(2, 9, 5, 7), k=2)
    assert [d.page_content for d in out] == ["chunk 9", "chunk 7"]


def test_rerank_empty_input_skips_model(monkeypatch):
    def boom():
        raise AssertionError("model should not load for empty input")
    monkeypatch.setattr(retrieval, "get_reranker", boom)
    assert retrieval.rerank("q", [], k=3) == []


def test_retrieve_reranks_candidate_pool_not_just_k(monkeypatch):
    calls = {}

    def fake_first_stage(query, doc_id, use_v2, k, mode):
        calls["k"] = k
        return _docs(1, 2, 3, 4)

    monkeypatch.setattr(retrieval, "_first_stage", fake_first_stage)
    monkeypatch.setattr(retrieval, "get_reranker", lambda: _FakeCrossEncoder())
    out = retrieval.retrieve("q", "d", use_v2=True, k=2, mode="hybrid", use_rerank=True)
    assert calls["k"] == retrieval.settings.candidate_k  # pool of 40, not 2
    assert [d.page_content for d in out] == ["chunk 4", "chunk 3"]


def test_retrieve_without_rerank_never_touches_model(monkeypatch):
    monkeypatch.setattr(retrieval, "_first_stage", lambda *a, **kw: _docs(1, 2, 3))
    monkeypatch.setattr(retrieval, "get_reranker",
                        lambda: (_ for _ in ()).throw(AssertionError("loaded")))
    out = retrieval.retrieve("q", "d", use_v2=True, k=3, mode="dense", use_rerank=False)
    assert len(out) == 3
