"""Naive-baseline eval harness: runs evals/golden.jsonl through the live
QueryService pipeline and scores answers with an LLM judge.

Usage:
    uv run python evals/run_eval.py
"""
from app.utils.qdrant import COLLECTION_NAME, COLLECTION_NAME_v2, client as qdrant_client
from app.utils.llm_factory import llm
from app.services.query import QueryService
from app.services.documents import DocumentsService
from app.schemas.query import QueryPayload
from app.repositories.documents import DocumentsRepository
from app.core.database import engine
import asyncio
import json
import sys
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


async def run_question(session: Session, item: dict, name_to_doc_id: dict, use_v2: bool = False) -> dict:
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

    answers = []
    for doc_id in target_doc_ids:
        service = QueryService(session)
        result = await service.query(QueryPayload(query=item["question"], document_id=doc_id, use_v2=use_v2))
        answers.append(result["answer"])

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

    return {
        "id": item["id"],
        "type": qtype,
        "cross_document": cross_document,
        "question": item["question"],
        "gold_answer": item["answer"],
        "model_answer": answers,
        "passed": passed,
        "judge_detail": detail,
    }


async def main():
    use_v2 = "--v2" in sys.argv
    RESULTS_DIR.mkdir(exist_ok=True)
    golden = load_golden()

    with Session(engine) as session:
        print(f"Checking ingestion status ({'v2/docling' if use_v2 else 'v1/naive'})...")
        name_to_doc_id = ensure_ingested(session, use_v2=use_v2)
        print(f"  {len(name_to_doc_id)} document(s) ready: {
              list(name_to_doc_id)}")

        results = []
        for i, item in enumerate(golden, 1):
            print(f"[{i}/{len(golden)}] {item['id']} ({item['type']})...")
            try:
                result = await run_question(session, item, name_to_doc_id, use_v2=use_v2)
            except Exception as exc:
                result = {
                    "id": item["id"], "type": item["type"], "cross_document": False,
                    "question": item["question"], "gold_answer": item["answer"],
                    "model_answer": None, "passed": False, "judge_detail": f"ERROR: {exc}",
                }
            results.append(result)
            print(
                f"  -> {'PASS' if result['passed'] else 'FAIL'}: {result['judge_detail'][:100]}")

    report = summarize(results)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_v2" if use_v2 else ""
    out_path = RESULTS_DIR / f"{ts}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"\nFull report written to {out_path}")


def summarize(results: list[dict]) -> dict:
    by_type = {}
    for r in results:
        key = r["type"] + ("_cross_document" if r["cross_document"] else "")
        by_type.setdefault(key, []).append(r)

    breakdown = {
        key: {
            "total": len(items),
            "passed": sum(1 for i in items if i["passed"]),
            "accuracy": round(sum(1 for i in items if i["passed"]) / len(items), 3) if items else None,
        }
        for key, items in by_type.items()
    }
    overall = {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "accuracy": round(sum(1 for r in results if r["passed"]) / len(results), 3) if results else None,
    }
    return {"overall": overall, "by_type": breakdown, "results": results}


def print_summary(report: dict):
    print("\n" + "=" * 50)
    print("EVAL SUMMARY")
    print("=" * 50)
    o = report["overall"]
    print(f"Overall: {o['passed']}/{o['total']} ({o['accuracy']:.1%})")
    print("-" * 50)
    for key, s in sorted(report["by_type"].items()):
        print(f"{key:30s} {s['passed']:2d}/{s['total']
              :2d}  ({s['accuracy']:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
