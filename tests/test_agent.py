import asyncio
import json

from langchain_core.documents import Document

from app.services import agent


def _doc(idx: int, text: str = "", pages=(1,), block_types=("text",)) -> Document:
    return Document(page_content=text or f"chunk {idx}",
                    metadata={"doc_id": "d", "chunk_index": idx, "pages": list(pages),
                              "block_types": list(block_types)})


class _Reply:
    def __init__(self, content: str):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 2}


class _FakeLLM:
    """Replies from a script, one entry per call, and records the prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return _Reply(self.replies.pop(0))


# ---- parsing ---------------------------------------------------------------

def test_parse_plan_reads_action_and_argument():
    raw = json.dumps({"sufficient": False, "missing": "Article 10", "action": "search",
                      "argument": "Article 10 data governance"})
    assert agent.parse_plan(raw) == ("SEARCH", "Article 10 data governance", "Article 10")


def test_parse_plan_sufficient_or_garbage_means_answer():
    assert agent.parse_plan('{"sufficient": true, "action": "SEARCH"}')[0] == "ANSWER"
    assert agent.parse_plan("not json")[0] == "ANSWER"
    assert agent.parse_plan('{"action": "BROWSE_WEB"}')[0] == "ANSWER"


def test_split_claims_extracts_citations_and_skips_fragments():
    answer = ("Article 5 lists prohibited practices [c3]. Article 10 sets data "
              "rules [c7, c9].\nThe two articles are:\n- It is fined heavily.")
    assert agent.split_claims(answer) == [
        ("Article 5 lists prohibited practices.", ["c3"]),
        ("Article 10 sets data rules.", ["c7", "c9"]),
        ("It is fined heavily.", []),
    ]


# ---- actions ---------------------------------------------------------------

def test_zoom_returns_neighbours_then_same_page(monkeypatch):
    chunks = [_doc(0, pages=[1]), _doc(1, pages=[2]), _doc(2, pages=[2]),
              _doc(3, pages=[2]), _doc(4, pages=[3])]
    monkeypatch.setattr(agent, "chunks_for", lambda doc_id, use_v2: chunks)
    out = [d.metadata["chunk_index"] for d in agent.zoom("d", "c2")]
    assert out[:2] == [1, 3]
    assert 0 not in out and 4 not in out
    assert agent.zoom("d", "c99") == [] and agent.zoom("d", "nothing") == []


def test_lookup_table_prefers_the_table_chunk_over_prose(monkeypatch):
    chunks = [
        _doc(0, "As Table 3 shows, complexity varies."),
        _doc(1, "Table 3: Comparison of architectures", block_types=("caption",)),
        _doc(2, "| Arch | Complexity |", block_types=("table",)),
        _doc(3, "Table 30 is unrelated", block_types=("table",)),
    ]
    monkeypatch.setattr(agent, "chunks_for", lambda doc_id, use_v2: chunks)
    out = [d.metadata["chunk_index"] for d in agent.lookup_table("d", "table 3")]
    assert out == [1, 2, 0]


# ---- verifier --------------------------------------------------------------

def test_verify_checks_uncited_claims_against_all_evidence():
    evidence = agent.Evidence()
    evidence.add([_doc(1, "Article 5 prohibits social scoring."), _doc(2, "Fines reach 30 million.")], limit=5)
    llm = _FakeLLM(['{"supported": true}', '{"supported": true}', '{"supported": false}'])
    trace = agent.Trace(mode="test")
    report = asyncio.run(agent.verify(
        "Article 5 prohibits social scoring [c1]. Fines reach 30 million. It also bans cars [c42].",
        evidence, llm, trace))
    assert [c["scope"] for c in report["claims"]] == ["cited", "all_evidence", "all_evidence"]
    assert [c["supported"] for c in report["claims"]] == [True, True, False]
    assert not report["passed"]
    assert trace.llm_calls == {"verify": 3}
    # the uncited claim was shown every gathered chunk, the cited one only c1
    assert "Fines reach" not in llm.prompts[0] and "Fines reach" in llm.prompts[1]


def test_uncited_but_supported_answer_passes(monkeypatch):
    answer, _, trace = _run_loop(
        monkeypatch, ['{"sufficient": true}'], ["The fine is 30 million euros."],
        ['{"supported": true}'])
    assert answer == "The fine is 30 million euros."
    assert trace["verification"]["passed"]


def test_strip_trailing_abstention_keeps_answer_and_pure_refusals():
    mixed = "The fine is 30 million euros. I don't have enough information in this document to answer that."
    assert agent.strip_trailing_abstention(mixed) == "The fine is 30 million euros."
    assert agent.strip_trailing_abstention(agent.NO_ANSWER) == agent.NO_ANSWER


# ---- the loop --------------------------------------------------------------

def _run_loop(monkeypatch, planner_replies, writer_replies, verifier_replies, max_hops=3):
    llms = {"json256": _FakeLLM(planner_replies), "json128": _FakeLLM(verifier_replies),
            "text": _FakeLLM(writer_replies)}

    def fake_build_llm(num_ctx=8192, num_predict=3072, timeout=180, format=None):
        return llms["text"] if format is None else llms[f"json{num_predict}"]

    counter = iter(range(100))
    monkeypatch.setattr(agent, "build_llm", fake_build_llm)
    monkeypatch.setattr(agent, "execute", lambda action, arg, q, doc_id: [_doc(next(counter), f"{action} {arg}")])
    monkeypatch.setattr(agent.settings, "agent_max_hops", max_hops)
    monkeypatch.setattr(agent.settings, "verify_repair_attempts", 1)
    return asyncio.run(agent.answer_agentic("question?", "d"))


def test_loop_stops_at_hop_budget_even_if_planner_keeps_going(monkeypatch):
    plan = lambda q: json.dumps({"sufficient": False, "action": "SEARCH", "argument": q})
    answer, docs, trace = _run_loop(
        monkeypatch, [plan("a"), plan("b"), plan("c"), plan("d")],
        ["The answer is here [c0]."], ['{"supported": true}'], max_hops=3)
    assert trace["hops"] == 3
    assert trace["llm_calls_by_stage"] == {"plan": 2, "answer": 1, "verify": 1}
    assert answer == "The answer is here [c0]."
    assert trace["input_tokens"] == 40 and trace["output_tokens"] == 8


def test_loop_stops_on_repeated_action(monkeypatch):
    plan = json.dumps({"sufficient": False, "action": "ZOOM", "argument": "c0"})
    _, _, trace = _run_loop(monkeypatch, [plan, plan], ["Done [c0]."], ['{"supported": true}'], max_hops=5)
    assert trace["hops"] == 2
    assert trace["actions"][-1]["stopped"] == "repeated action"


def test_unsupported_answer_is_repaired_once_then_rejected(monkeypatch):
    answer, _, trace = _run_loop(
        monkeypatch, ['{"sufficient": true}'],
        ["Made up claim here [c0].", "Still made up here [c0]."],
        ['{"supported": false}', '{"supported": false}'])
    assert answer == agent.NO_ANSWER
    assert trace["verification"]["repair_attempts"] == 1
    assert trace["verification"]["rejected_answer"] == "Still made up here [c0]."
