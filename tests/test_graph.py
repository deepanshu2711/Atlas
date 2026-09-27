from langchain_core.documents import Document

from app.services import retrieval
from app.services.graph_extract import parse_extraction
from app.utils.graph_store import ChunkGraph, GraphStore, normalize


def _store(tmp_path) -> GraphStore:
    return GraphStore(str(tmp_path / "graph.kuzu"))


def _sample_chunks() -> list[ChunkGraph]:
    return [
        ChunkGraph(0, entities=[{"name": "AI Act", "type": "regulation"},
                                {"name": "high-risk AI system", "type": "concept"}],
                   relations=[{"source": "AI Act", "target": "high-risk AI system",
                               "type": "regulates"}]),
        ChunkGraph(1, entities=[{"name": "High-Risk AI System", "type": "concept"},
                                {"name": "conformity assessment", "type": "process"}],
                   relations=[{"source": "high-risk AI system",
                               "target": "conformity assessment", "type": "requires"}]),
        ChunkGraph(2, entities=[{"name": "conformity assessment", "type": "process"}]),
        ChunkGraph(3, entities=[{"name": "GDPR", "type": "regulation"}]),
    ]


def test_normalize_merges_case_and_punctuation():
    assert normalize("The  AI-Act") == normalize("the ai act") == "the ai act"


def test_parse_extraction_reads_clean_json():
    entities, relations = parse_extraction(
        '{"entities": [{"name": " AI Act ", "type": "law"}],'
        ' "relations": [{"source": "AI Act", "target": "EU", "type": "adopted_by"}]}')
    assert entities == [{"name": "AI Act", "type": "law"}]
    assert relations == [{"source": "AI Act", "target": "EU", "type": "adopted_by"}]


def test_parse_extraction_tolerates_fences_and_junk():
    raw = 'Sure!\n```json\n{"entities": [{"name": "GDPR"}, {"type": "no name"}, "x"],' \
          ' "relations": [{"source": "GDPR"}]}\n```'
    entities, relations = parse_extraction(raw)
    assert entities == [{"name": "GDPR", "type": ""}]
    assert relations == []  # missing target


def test_parse_extraction_gives_up_quietly():
    assert parse_extraction("not json at all") == ([], [])
    assert parse_extraction("[1, 2]") == ([], [])


def test_match_entities_needs_whole_words(tmp_path):
    store = _store(tmp_path)
    store.replace_document("d", _sample_chunks())
    matched = store.match_entities("d", "What does the AI Act say about GDPR?")
    assert sorted(matched) == ["d:ai act", "d:gdpr"]
    # "gdpr" inside another word is not a mention
    assert store.match_entities("d", "What about gdprx?") == []


def test_chunk_scores_counts_mentions_and_one_hop(tmp_path):
    store = _store(tmp_path)
    store.replace_document("d", _sample_chunks())
    scores = store.chunk_scores(["d:ai act"])
    # chunk 0 mentions AI Act (1) and its neighbour high-risk AI system (0.5);
    # chunk 1 only mentions the neighbour; chunks 2 and 3 are two hops or more away
    assert scores == {0: 1.5, 1: 0.5}


def test_replace_document_is_idempotent_and_scoped(tmp_path):
    store = _store(tmp_path)
    store.replace_document("d", _sample_chunks())
    store.replace_document("other", [ChunkGraph(0, entities=[{"name": "AI Act"}])])
    assert store.replace_document("d", _sample_chunks()) == 4  # same graph, no duplicates
    assert store.chunk_scores(["d:ai act"]) == {0: 1.5, 1: 0.5}
    assert store.has_document("other")
    assert store.match_entities("other", "the AI Act") == ["other:ai act"]
    store.replace_document("d", [])
    assert not store.has_document("d")


def _doc(idx: int) -> Document:
    return Document(page_content=f"chunk {idx}", metadata={"doc_id": "d", "chunk_index": idx})


def test_hybrid_graph_fuses_graph_ranking(monkeypatch):
    monkeypatch.setattr(retrieval, "dense_search", lambda *a: [_doc(1), _doc(2)])
    monkeypatch.setattr(retrieval, "bm25_search", lambda *a: [_doc(2), _doc(1)])
    monkeypatch.setattr(retrieval, "graph_search", lambda *a: [_doc(3), _doc(1)])
    out = retrieval._first_stage("q", "d", True, 3, "hybrid_graph")
    # chunk 1 is in all three rankings; chunk 3 only in the graph one
    assert [d.metadata["chunk_index"] for d in out] == [1, 2, 3]


def test_plain_hybrid_never_touches_graph(monkeypatch):
    monkeypatch.setattr(retrieval, "dense_search", lambda *a: [_doc(1)])
    monkeypatch.setattr(retrieval, "bm25_search", lambda *a: [_doc(2)])
    monkeypatch.setattr(retrieval, "graph_search",
                        lambda *a: (_ for _ in ()).throw(AssertionError("graph used")))
    assert len(retrieval._first_stage("q", "d", True, 3, "hybrid")) == 2
