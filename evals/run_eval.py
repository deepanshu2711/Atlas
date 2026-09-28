"""Naive-baseline eval harness: runs evals/golden.jsonl through the live
QueryService pipeline and scores answers with an LLM judge.

Usage:
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --v2 --retrieval-only [--mode dense|bm25|hybrid|graph|hybrid_graph] [--rerank]   # no LLM; page recall@k / MRR
    uv run python evals/run_eval.py --v2 --agentic   # assess/plan/execute loop + citation verifier
    uv run python evals/compare.py <baseline.json> <candidate.json>   # accuracy gained vs cost added
"""
from app.utils.qdrant import COLLECTION_NAME, COLLECTION_NAME_v2, client as qdrant_client
from app.utils.llm_factory import llm
from app.core.config import settings
from app.services.query import QueryService
from app.utils.reranker import get_reranker
from app.services.documents import DocumentsService
from app.services.bm25 import chunks_for
from app.services.graph import index_document as index_graph
from app.utils.graph_store import get_graph_store
from app.schemas.query import QueryPayload
from app.repositories.documents import DocumentsRepository
from app.core.database import engine
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlmodel import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

NO_CONTEXT_MESSAGE = "I don't have enough information in this document to answer that."

JUDGE_PROMPT = """You are grading a RAG system's answer against a gold answer.

Question: {question}
Gold answer: {gold_answer}
Question type: {qtype}
System's answer: {model_answer}

{instructions}

Respond with exactly one word on the first line: {labels}
Then a one-sentence reason on the second line."""

INSTRUCTIONS_NORMAL = (
    "Judge whether the system's answer is factually consistent with the gold "
    "answer (same facts/entities/numbers - wording can differ)."
)
LABELS_NORMAL = "CORRECT or INCORRECT"

INSTRUCTIONS_UNANSWERABLE = (
    "The gold answer states this question cannot be answered from the corpus. "
    "Judge whether the system's answer likewise declines / says it doesn't "
    "have enough information (ABSTAINED), or whether it confidently states a "
    "specific answer anyway (HALLUCINATED)."
)
LABELS_UNANSWERABLE = "ABSTAINED or HALLUCINATED"


def arg_value(flag: str) -> str | None:
    """Value following `flag` on the command line, e.g. --mode hybrid."""
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def load_golden():
    with open(GOLDEN_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_ingested(session: Session, use_v2: bool = False):
    """Make sure every doc referenced in docs/ has vectors in Qdrant.
    Ingestion is idempotent (deterministic chunk ids), so safe to re-run."""
    collection = COLLECTION_NAME_v2 if use_v2 else COLLECTION_NAME
    ingest_fn = DocumentsService.ingest_document_v2 if use_v2 else DocumentsService.ingest_document

    repo = DocumentsRepository(session)
    name_to_doc_id = {}
    for document in repo.all():
        count = qdrant_client.count(
            collection_name=collection,
            count_filter=Filter(must=[
                FieldCondition(key="metadata.doc_id",
                               match=MatchValue(value=document.doc_id))
            ]),
        ).count
        if count == 0:
            print(f"  re-ingesting {document.name} ({document.doc_id}) into {collection}...")
            try:
                ingest_fn(document.doc_id, document.file_path)
                document.status = "ready"
            except Exception as exc:
                document.status = "failed"
                print(f"  FAILED to ingest {document.name}: {exc}")
            repo.update(document)
        name_to_doc_id[document.name] = document.doc_id
    return name_to_doc_id


def ensure_graph_indexed(name_to_doc_id: dict):
    """Build the Kuzu graph for any document that has v2 chunks but no graph
    yet (ingested before graph_enabled). Reads the chunks back from Qdrant,
    so nothing is re-parsed; costs one LLM call per chunk."""
    store = get_graph_store()
    for name, doc_id in name_to_doc_id.items():
        if store.has_document(doc_id):
            continue
        chunks = chunks_for(doc_id, use_v2=True)
        print(f"  building graph for {name} ({len(chunks)} chunks)...")
        n = index_graph(doc_id, [(d.metadata["chunk_index"], d.page_content) for d in chunks])
        print(f"    {n} entities")


async def judge(question, gold_answer, qtype, model_answer, unanswerable: bool) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        qtype=qtype,
        model_answer=model_answer,
        instructions=INSTRUCTIONS_UNANSWERABLE if unanswerable else INSTRUCTIONS_NORMAL,
        labels=LABELS_UNANSWERABLE if unanswerable else LABELS_NORMAL,
    )
    response = await llm.ainvoke(prompt)
    text = response.content.strip()
    first_line = text.splitlines()[0].strip().upper() if text else ""
    # Order matters: "INCORRECT" contains "CORRECT" as a substring, so the
    # more specific label must be checked first (a set's iteration order is
    # randomized per-process and previously made this check non-deterministic).
    ordered_labels = ["HALLUCINATED", "ABSTAINED"] if unanswerable else [
        "INCORRECT", "CORRECT"]
    verdict = next(
        (v for v in ordered_labels if v in first_line), "UNPARSEABLE")
    return verdict, text


COST_KEYS = ("hops", "llm_calls", "input_tokens", "output_tokens", "latency_s")


async def run_question(session: Session, item: dict, name_to_doc_id: dict, use_v2: bool = False,
                       agentic: bool = False) -> dict:
    qtype = item["type"]
    unanswerable = qtype == "unanswerable"
    doc_field = item["doc"]

    if unanswerable:
        target_doc_ids = list(name_to_doc_id.values())
        cross_document = False
    elif isinstance(doc_field, list):
        # Current QueryService is single-document-scoped; cross-document
        # multi-hop questions are out of scope for this architecture.
        # Best-effort: query the first referenced doc only, flagged separately.
        target_doc_ids = [name_to_doc_id[doc_field[0]]]
        cross_document = True
    else:
        target_doc_ids = [name_to_doc_id[doc_field]]
        cross_document = False

    answers, traces, sources = [], [], []
    for doc_id in target_doc_ids:
        service = QueryService(session)
        result = await service.query(QueryPayload(query=item["question"], document_id=doc_id,
                                                  use_v2=use_v2, agentic=agentic))
        answers.append(result["answer"])
        traces.append(result["trace"])
        sources.extend(result["sources"])

    if unanswerable:
        # must abstain on every document in the corpus to count as correct
        verdicts = []
        for ans in answers:
            v, _ = await judge(item["question"], item["answer"], qtype, ans, unanswerable=True)
            verdicts.append(v)
        passed = all(v == "ABSTAINED" for v in verdicts)
        detail = "; ".join(verdicts)
    else:
        combined_answer = " | ".join(answers)
        verdict, detail = await judge(item["question"], item["answer"], qtype, combined_answer, unanswerable=False)
        passed = verdict == "CORRECT"

    # Cost of answering (the judge is not part of the system, so excluded);
    # an unanswerable item queries every document, so its cost is the sum.
    cost = {key: round(sum(t[key] for t in traces), 3) for key in COST_KEYS}
    verifications = [t["verification"] for t in traces if t["verification"] is not None]
    evidence = {}
    if not unanswerable and use_v2:
        gold = gold_pages_for_queried_doc(item)
        pages = {p for d in sources for p in d.metadata.get("pages", [])}
        evidence = {"evidence_pages": sorted(pages),
                    "evidence_recall": len(set(gold) & pages) / len(gold)}

    return {
        "id": item["id"],
        "type": qtype,
        "cross_document": cross_document,
        "question": item["question"],
        "gold_answer": item["answer"],
        "model_answer": answers,
        "passed": passed,
        "judge_detail": detail,
        **cost,
        **evidence,
        "verified": all(v["passed"] for v in verifications) if verifications else None,
        "trace": traces,
    }


RETRIEVAL_KS = (3, 8, 40)


def gold_pages_for_queried_doc(item: dict) -> list[int]:
    """Gold pages that live in the document QueryService is asked about.
    Cross-document items only query the first doc, so only its page counts."""
    pages = item["page"] if isinstance(item["page"], list) else [item["page"]]
    if isinstance(item["doc"], list):
        return pages[:1]
    return pages


def score_retrieval(gold_pages: list[int], docs) -> dict:
    """Page-level metrics. A chunk is a hit for every gold page it spans."""
    gold = set(gold_pages)
    out = {}
    for k in RETRIEVAL_KS:
        covered = set()
        for d in docs[:k]:
            covered |= gold & set(d.metadata.get("pages", []))
        out[f"recall@{k}"] = len(covered) / len(gold)
        out[f"hit@{k}"] = 1.0 if covered else 0.0
    out["mrr"] = next(
        (1 / rank for rank, d in enumerate(docs, 1)
         if gold & set(d.metadata.get("pages", []))), 0.0)
    return out


def run_retrieval_only(session: Session, golden: list[dict], name_to_doc_id: dict, mode=None, use_rerank=False) -> dict:
    per_question = []
    latencies = []
    if use_rerank:
        get_reranker()  # load the model up front so it isn't counted as query latency
    for item in golden:
        if item["type"] == "unanswerable":
            continue
        doc_name = item["doc"][0] if isinstance(item["doc"], list) else item["doc"]
        payload = QueryPayload(query=item["question"],
                               document_id=name_to_doc_id[doc_name], use_v2=True)
        start = time.perf_counter()
        docs = QueryService(session).retrieve(
            payload, k=max(RETRIEVAL_KS), mode=mode, use_rerank=use_rerank)
        latencies.append(time.perf_counter() - start)
        gold = gold_pages_for_queried_doc(item)
        per_question.append({
            "id": item["id"], "type": item["type"],
            "cross_document": isinstance(item["doc"], list),
            "gold_pages": gold,
            "retrieved_pages": [d.metadata.get("pages", []) for d in docs[:8]],
            **score_retrieval(gold, docs),
        })

    metric_keys = [f"{m}@{k}" for k in RETRIEVAL_KS for m in ("recall", "hit")] + ["mrr"]

    def mean(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None

    by_type = {}
    for r in per_question:
        by_type.setdefault(r["type"] + ("_cross_document" if r["cross_document"] else ""), []).append(r)
    return {
        "overall": {"total": len(per_question), **{m: mean(per_question, m) for m in metric_keys},
                    "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None},
        "by_type": {t: {"total": len(rows), **{m: mean(rows, m) for m in metric_keys}}
                    for t, rows in by_type.items()},
        "results": per_question,
    }


def print_retrieval_summary(report: dict):
    print("\n" + "=" * 72)
    print("RETRIEVAL SUMMARY (page-level)")
    print("=" * 72)
    cols = [f"R@{k}" for k in RETRIEVAL_KS] + [f"H@{k}" for k in RETRIEVAL_KS] + ["MRR"]
    keys = [f"recall@{k}" for k in RETRIEVAL_KS] + [f"hit@{k}" for k in RETRIEVAL_KS] + ["mrr"]
    print(f"mean retrieval latency: {report['overall']['mean_latency_s']}s/query")
    print(f"{'':32s}{'n':>3s} " + " ".join(f"{c:>6s}" for c in cols))
    rows = [("overall", report["overall"])] + sorted(report["by_type"].items())
    for name, s in rows:
        print(f"{name:32s}{s['total']:3d} " + " ".join(f"{s[k]:6.3f}" for k in keys))


async def main():
    retrieval_only = "--retrieval-only" in sys.argv
    agentic = "--agentic" in sys.argv
    # only v2 chunks carry page metadata, and the agentic loop needs them
    use_v2 = "--v2" in sys.argv or retrieval_only or agentic
    RESULTS_DIR.mkdir(exist_ok=True)
    golden = load_golden()

    with Session(engine) as session:
        print(f"Checking ingestion status ({'v2/docling' if use_v2 else 'v1/naive'})...")
        name_to_doc_id = ensure_ingested(session, use_v2=use_v2)
        print(f"  {len(name_to_doc_id)} document(s) ready: {
              list(name_to_doc_id)}")

        # --mode only applies to --retrieval-only; otherwise settings decide
        mode = (arg_value('--mode') if retrieval_only else None) or settings.retrieval_mode
        if use_v2 and mode in ("graph", "hybrid_graph"):
            print("Checking graph index...")
            ensure_graph_indexed(name_to_doc_id)

        if retrieval_only:
            report = run_retrieval_only(session, golden, name_to_doc_id, mode=arg_value('--mode'),
                                        use_rerank='--rerank' in sys.argv)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = RESULTS_DIR / f"{ts}_retrieval_{arg_value('--mode') or 'default'}{'_rerank' if '--rerank' in sys.argv else ''}.json"
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print_retrieval_summary(report)
            print(f"\nFull report written to {out_path}")
            return

        results = []
        for i, item in enumerate(golden, 1):
            print(f"[{i}/{len(golden)}] {item['id']} ({item['type']})...")
            try:
                result = await run_question(session, item, name_to_doc_id, use_v2=use_v2, agentic=agentic)
            except Exception as exc:
                result = {
                    "id": item["id"], "type": item["type"], "cross_document": False,
                    "question": item["question"], "gold_answer": item["answer"],
                    "model_answer": None, "passed": False, "judge_detail": f"ERROR: {exc}",
                }
            results.append(result)
            cost = f" [hops={result['hops']} calls={result['llm_calls']} " \
                f"tok={result['input_tokens'] + result['output_tokens']} {result['latency_s']}s]" \
                if "hops" in result else ""
            print(
                f"  -> {'PASS' if result['passed'] else 'FAIL'}{cost}: {result['judge_detail'][:100]}")

    report = summarize(results)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_v2" if use_v2 else ""
    if use_v2:  # record which retrieval config produced this run
        cfg = {"mode": settings.retrieval_mode, "rerank": settings.rerank_enabled,
               "final_k": settings.final_k, "num_ctx": llm.num_ctx, "agentic": agentic}
        if agentic:
            cfg.update(max_hops=settings.agent_max_hops, step_k=settings.agent_step_k,
                       agent_num_ctx=settings.agent_num_ctx, verify=settings.verify_enabled,
                       verify_repair_attempts=settings.verify_repair_attempts)
        report["config"] = cfg
        suffix += f"_{cfg['mode']}{'_rerank' if cfg['rerank'] else ''}_k{cfg['final_k']}"
        if agentic:
            suffix += f"_agentic_h{settings.agent_max_hops}{'' if settings.verify_enabled else '_noverify'}"
    out_path = RESULTS_DIR / f"{ts}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"\nFull report written to {out_path}")


def _stats(items: list[dict]) -> dict:
    """Accuracy plus mean cost per question. Errored items have no cost
    fields and are left out of the means."""
    def mean(key):
        values = [i[key] for i in items if i.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    out = {
        "total": len(items),
        "passed": sum(1 for i in items if i["passed"]),
        "accuracy": round(sum(1 for i in items if i["passed"]) / len(items), 3) if items else None,
    }
    for key in COST_KEYS + ("evidence_recall",):
        out[f"mean_{key}"] = mean(key)
    costed = [i for i in items if "input_tokens" in i]
    out["mean_tokens"] = round(sum(i["input_tokens"] + i["output_tokens"] for i in costed)
                               / len(costed), 1) if costed else None
    verified = [i["verified"] for i in items if i.get("verified") is not None]
    out["verify_pass_rate"] = round(sum(verified) / len(verified), 3) if verified else None
    return out


def summarize(results: list[dict]) -> dict:
    by_type = {}
    for r in results:
        key = r["type"] + ("_cross_document" if r["cross_document"] else "")
        by_type.setdefault(key, []).append(r)

    breakdown = {key: _stats(items) for key, items in by_type.items()}
    return {"overall": _stats(results), "by_type": breakdown, "results": results}


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def print_summary(report: dict):
    print("\n" + "=" * 90)
    print("EVAL SUMMARY")
    print("=" * 90)
    print(f"{'':30s}{'pass':>7s} {'acc':>6s} {'hops':>5s} {'calls':>5s} "
          f"{'tokens':>7s} {'lat_s':>6s} {'ev_rec':>6s} {'verif':>6s}")
    rows = [("overall", report["overall"])] + sorted(report["by_type"].items())
    for key, s in rows:
        print(f"{key:30s}{s['passed']:3d}/{s['total']:<3d} {s['accuracy']:6.1%} "
              f"{_fmt(s['mean_hops'], '5.2f')} {_fmt(s['mean_llm_calls'], '5.2f')} "
              f"{_fmt(s['mean_tokens'], '7.0f')} {_fmt(s['mean_latency_s'], '6.2f')} "
              f"{_fmt(s['mean_evidence_recall'], '6.3f')} {_fmt(s['verify_pass_rate'], '6.3f')}")


if __name__ == "__main__":
    asyncio.run(main())
