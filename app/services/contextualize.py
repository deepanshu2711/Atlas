from langchain_ollama import ChatOllama
from app.core.logging import get_logger
from app.utils.llm_factory import build_llm

logger = get_logger(__name__)

SUMMARY_SAMPLE_CHARS = 1200

SUMMARY_PROMPT = """Here is an excerpt from the beginning of a document:

{sample_text}

In one sentence, summarize what this document is about. Respond with only the sentence."""

SITUATE_PROMPT = """Document summary: {doc_summary}
Section: {heading_path}

Chunk:
{chunk_text}

In 1-2 short sentences (max ~40 words), describe what this chunk covers and how \
it fits into the document, so someone reading only this note understands the \
chunk without the rest of the document. Respond with only the sentences."""


def build_context_llm() -> ChatOllama:
    # Small, fast: short prompts, short outputs - not the query-time llm singleton.
    return build_llm(num_ctx=2048, num_predict=80, timeout=60)


def summarize_document(llm: ChatOllama, sample_text: str) -> str:
    try:
        response = llm.invoke(
            SUMMARY_PROMPT.format(sample_text=sample_text[:SUMMARY_SAMPLE_CHARS]))
        return response.content.strip()
    except Exception:
        logger.exception("summarize_document failed")
        return ""


def situate_chunk(llm: ChatOllama, doc_summary: str, heading_path: list[str], chunk_text: str) -> str:
    try:
        response = llm.invoke(SITUATE_PROMPT.format(
            doc_summary=doc_summary or "(no summary available)",
            heading_path=" > ".join(heading_path or []) or "(no heading)",
            chunk_text=chunk_text,
        ))
        return response.content.strip()
    except Exception:
        logger.exception("situate_chunk failed")
        return ""
