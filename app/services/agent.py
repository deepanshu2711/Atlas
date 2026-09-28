"""Agentic retrieval: an assess/plan/execute loop over one document with a
hard hop budget, followed by a citation verifier that fails the answer when
any claim is not supported by the chunk it cites.

Every LLM call goes through `Trace`, so a caller (the eval harness) can put
hops, calls, tokens and latency next to accuracy and decide whether the loop
is worth what it costs over single-pass retrieval.
"""
import json
import re
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.core.logging import get_logger
from app.services.bm25 import chunks_for
from app.services.retrieval import rerank, retrieve
from app.utils.llm_factory import build_llm

logger = get_logger(__name__)

NO_ANSWER = "I don't have enough information in this document to answer that."
ACTIONS = ("SEARCH", "ZOOM", "LOOKUP_TABLE", "ANSWER")
SNIPPET_CHARS = 300

# Literal braces are doubled because the prompts go through str.format.
PLAN_PROMPT = """You are gathering evidence from one document to answer a question.

Question: {question}

Evidence gathered so far (id, pages, heading, start of text):
{evidence}

Step 1 - assess: can the question be fully answered from this evidence? If not, what exactly is missing?
Step 2 - plan: choose ONE next action.
- SEARCH: search the document again. argument = a short, specific search query for the missing piece (not the whole question).
- ZOOM: read the chunks around an evidence chunk that looks relevant but cut off. argument = its id, e.g. "c12".
- LOOKUP_TABLE: fetch a table by name. argument = e.g. "Table 3".
- ANSWER: the evidence is enough. argument = "".

Respond with JSON only, in exactly this shape:
{{"sufficient": true or false, "missing": "...", "action": "SEARCH|ZOOM|LOOKUP_TABLE|ANSWER", "argument": "..."}}"""

ANSWER_PROMPT = """Answer the question using ONLY the evidence below. Each evidence chunk starts with its id in brackets, e.g. [c12].

Rules:
- After every sentence, cite the id(s) of the chunk(s) that support it, e.g. "Article 5 prohibits social scoring [c12]."
- Do not state anything the cited chunks do not say. Do not use outside knowledge.
- If the evidence does not answer the question, reply exactly: "{no_answer}"
- Be concise: answer the question directly first, then supporting detail.
{feedback}
Evidence:
{evidence}

Question: {question}"""

REPAIR_FEEDBACK = """- A previous draft made these claims, which the cited chunks do NOT support. Leave them out or cite the chunk that does support them:
{claims}
"""

VERIFY_PROMPT = """Does the source text support the claim? The claim is supported only if the source states it or it follows directly from the source.

Source text:
{sources}

Claim: {claim}

Respond with JSON only, in exactly this shape:
{{"supported": true or false, "reason": "..."}}"""


@dataclass
class Trace:
    """Cost and behaviour of answering one question."""
    mode: str
    hops: int = 0
    actions: list[dict] = field(default_factory=list)
    llm_calls: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    verification: dict | None = None
    _start: float = field(default_factory=time.perf_counter, repr=False)

    async def call(self, llm: ChatOllama, stage: str, prompt) -> str:
        response = await llm.ainvoke(prompt)
        self.llm_calls[stage] = self.llm_calls.get(stage, 0) + 1
        usage = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        return response.content

    def finish(self) -> dict:
        self.latency_s = round(time.perf_counter() - self._start, 3)
        return {
            "mode": self.mode,
            "hops": self.hops,
            "actions": self.actions,
            "llm_calls": sum(self.llm_calls.values()),
            "llm_calls_by_stage": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": self.latency_s,
            "verification": self.verification,
        }


def chunk_id(doc: Document) -> str:
    return f"c{doc.metadata['chunk_index']}"


class Evidence:
    """Chunks gathered so far, in the order they were found, deduplicated."""

    def __init__(self) -> None:
        self.docs: dict[str, Document] = {}

    def add(self, docs: list[Document], limit: int) -> list[str]:
        """Add up to `limit` chunks not already held; return their ids."""
        added = []
        for doc in docs:
            if len(added) == limit:
                break
            cid = chunk_id(doc)
            if cid not in self.docs:
                self.docs[cid] = doc
                added.append(cid)
        return added

    def snippets(self) -> str:
        if not self.docs:
            return "(none)"
        lines = []
        for cid, doc in self.docs.items():
            heading = " > ".join(doc.metadata.get("headings") or [])
            text = " ".join(doc.page_content.split())[:SNIPPET_CHARS]
            lines.append(f"[{cid}] pages={doc.metadata.get('pages', [])} heading={heading!r}: {text}")
        return "\n".join(lines)

    def full_text(self) -> str:
        return "\n\n---\n\n".join(f"[{cid}] {doc.page_content}" for cid, doc in self.docs.items())


# ---- actions ---------------------------------------------------------------

def zoom(doc_id: str, target: str) -> list[Document]:
    """Chunks next to `target` (e.g. "c12"): the one before and after it, then
    the rest of the chunks on the same pages."""
    match = re.search(r"\d+", target)
    if not match:
        return []
    idx = int(match.group())
    chunks = chunks_for(doc_id, use_v2=True)
    by_index = {d.metadata["chunk_index"]: d for d in chunks}
    if idx not in by_index:
        return []
    pages = set(by_index[idx].metadata.get("pages", []))
    neighbours = [by_index[i] for i in (idx - 1, idx + 1) if i in by_index]
    # closest first, since the caller keeps only the first few
    same_page = sorted(
        (d for d in chunks if abs(d.metadata["chunk_index"] - idx) > 1
         and pages & set(d.metadata.get("pages", []))),
        key=lambda d: abs(d.metadata["chunk_index"] - idx))
    return neighbours + same_page


def _has_table(doc: Document) -> bool:
    return any(label in ("table", "caption") for label in doc.metadata.get("block_types", []))


def lookup_table(doc_id: str, name: str) -> list[Document]:
    """Chunks holding the table called `name` (e.g. "Table 3").

    A chunk mentioning "Table 3" that also contains a table or caption block
    is almost certainly the table itself; prose that only refers to it comes
    after. A caption chunk's next chunk is included too, since the chunker can
    split a caption from its table. With no table number in `name`, fall back
    to ranking all table chunks against it."""
    chunks = chunks_for(doc_id, use_v2=True)
    number = re.search(r"table\s*([0-9]+|[ivxlc]+)\b", name, re.IGNORECASE)
    if not number:
        return rerank(name, [d for d in chunks if _has_table(d)], k=settings.agent_step_k)
    pattern = re.compile(rf"\btable\s+{re.escape(number.group(1))}\b", re.IGNORECASE)
    mentions = [d for d in chunks if pattern.search(d.page_content)]
    by_index = {d.metadata["chunk_index"]: d for d in chunks}
    out = []
    for doc in sorted(mentions, key=lambda d: not _has_table(d)):
        out.append(doc)
        following = by_index.get(doc.metadata["chunk_index"] + 1)
        if _has_table(doc) and following is not None and _has_table(following):
            out.append(following)
    return out


def execute(action: str, argument: str, question: str, doc_id: str) -> list[Document]:
    if action == "SEARCH":
        return retrieve(argument or question, doc_id, use_v2=True, k=settings.final_k)
    if action == "ZOOM":
        return zoom(doc_id, argument)
    if action == "LOOKUP_TABLE":
        return lookup_table(doc_id, argument)
    return []


# ---- planning --------------------------------------------------------------

def parse_json(raw: str) -> dict:
    """Parse the model's JSON reply, tolerating prose or fences around it."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            data = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def parse_plan(raw: str) -> tuple[str, str, str]:
    """(action, argument, missing); anything unparseable means ANSWER, so a
    confused planner ends the loop instead of burning the budget."""
    data = parse_json(raw)
    action = str(data.get("action") or "").strip().upper()
    argument = str(data.get("argument") or "").strip()
    missing = str(data.get("missing") or "").strip()
    if data.get("sufficient") is True or action not in ACTIONS:
        action = "ANSWER"
    return action, argument, missing


# ---- verification ----------------------------------------------------------

_CITATION_RE = re.compile(r"\[([^\]]*\bc\d+[^\]]*)\]")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_claims(answer: str) -> list[tuple[str, list[str]]]:
    """Split an answer into (claim, cited chunk ids). Sentences with too
    little content to be a claim (list intros, stray bullets) are skipped;
    everything else is a claim, cited or not."""
    claims = []
    for sentence in _SENTENCE_END_RE.split(answer):
        sentence = sentence.strip(" \t-*•")
        cited = [f"c{n}" for group in _CITATION_RE.findall(sentence)
                 for n in re.findall(r"c(\d+)", group)]
        text = re.sub(r"\s+([.,;!?])", r"\1", _CITATION_RE.sub("", sentence)).strip()
        if len(re.findall(r"\w+", text)) < 2 or text.endswith(":"):
            continue
        claims.append((text, list(dict.fromkeys(cited))))
    return claims


async def verify(answer: str, evidence: Evidence, llm: ChatOllama, trace: Trace) -> dict:
    """Check every claim against the chunks it cites. A claim with no
    citation, or citing a chunk that was never retrieved, fails without an
    LLM call."""
    results = []
    for claim, cited in split_claims(answer)[:settings.verify_max_claims]:
        sources = [evidence.docs[c] for c in cited if c in evidence.docs]
        if not sources:
            results.append({"claim": claim, "cited": cited, "supported": False,
                            "reason": "no citation" if not cited else "cites a chunk that was not retrieved"})
            continue
        raw = await trace.call(llm, "verify", VERIFY_PROMPT.format(
            sources="\n\n---\n\n".join(d.page_content for d in sources), claim=claim))
        data = parse_json(raw)
        results.append({"claim": claim, "cited": cited, "supported": data.get("supported") is True,
                        "reason": str(data.get("reason") or "")})
    return {"passed": all(r["supported"] for r in results), "claims": results}


# ---- the loop --------------------------------------------------------------

async def generate(question: str, evidence: Evidence, llm: ChatOllama, trace: Trace,
                   unsupported: list[str] | None = None) -> str:
    feedback = REPAIR_FEEDBACK.format(
        claims="\n".join(f"  - {c}" for c in unsupported)) if unsupported else ""
    return (await trace.call(llm, "answer", ANSWER_PROMPT.format(
        no_answer=NO_ANSWER, feedback=feedback, evidence=evidence.full_text(),
        question=question))).strip()


def is_abstention(answer: str) -> bool:
    return "don't have enough information" in answer.lower()


async def answer_agentic(question: str, doc_id: str) -> tuple[str, list[Document], dict]:
    """Run the loop and return (answer, evidence chunks, trace).

    Hop 1 is always a plain search on the question, so the loop starts from
    exactly what single-pass retrieval would have seen; the planner then gets
    up to `agent_max_hops - 1` further actions. The budget is enforced here,
    not left to the model."""
    trace = Trace(mode="agentic")
    planner = build_llm(num_ctx=settings.agent_num_ctx, num_predict=256, format="json")
    writer = build_llm(num_ctx=settings.agent_num_ctx)
    evidence = Evidence()

    evidence.add(execute("SEARCH", question, question, doc_id), limit=settings.final_k)
    trace.hops = 1
    trace.actions.append({"action": "SEARCH", "argument": question, "added": list(evidence.docs)})

    seen = {("SEARCH", question.lower())}
    while trace.hops < settings.agent_max_hops:
        action, argument, missing = parse_plan(await trace.call(
            planner, "plan", PLAN_PROMPT.format(question=question, evidence=evidence.snippets())))
        if action == "ANSWER":
            break
        if (action, argument.lower()) in seen:
            # repeating an action cannot find anything new; stop spending
            trace.actions.append({"action": action, "argument": argument, "missing": missing,
                                  "added": [], "stopped": "repeated action"})
            break
        seen.add((action, argument.lower()))
        added = evidence.add(execute(action, argument, question, doc_id), limit=settings.agent_step_k)
        trace.hops += 1
        trace.actions.append({"action": action, "argument": argument, "missing": missing, "added": added})

    if not evidence.docs:
        return NO_ANSWER, [], trace.finish()

    answer = await generate(question, evidence, writer, trace)
    if settings.verify_enabled and not is_abstention(answer):
        verifier = build_llm(num_ctx=settings.agent_num_ctx, num_predict=128, format="json")
        report = await verify(answer, evidence, verifier, trace)
        attempts = 0
        while not report["passed"] and attempts < settings.verify_repair_attempts:
            attempts += 1
            unsupported = [r["claim"] for r in report["claims"] if not r["supported"]]
            answer = await generate(question, evidence, writer, trace, unsupported=unsupported)
            if is_abstention(answer):
                break
            report = await verify(answer, evidence, verifier, trace)
        report["repair_attempts"] = attempts
        if not report["passed"] and not is_abstention(answer):
            report["rejected_answer"] = answer
            answer = NO_ANSWER
        trace.verification = report

    return answer, list(evidence.docs.values()), trace.finish()
