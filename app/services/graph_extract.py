import json

from langchain_ollama import ChatOllama

from app.core.logging import get_logger
from app.utils.graph_store import ChunkGraph
from app.utils.llm_factory import build_llm

logger = get_logger(__name__)

MAX_ENTITIES = 10
MAX_RELATIONS = 10

# Literal braces are doubled because the prompt goes through str.format.
EXTRACT_PROMPT = """Extract the key entities and the relationships between them from the text below.

Entities are specific things a reader might ask about: organisations, laws and \
regulations, articles or sections, systems, named concepts, roles, people, \
places, dates and metrics. Skip generic words like "system" or "document".

Respond with JSON only, in exactly this shape:
{{"entities": [{{"name": "...", "type": "..."}}],
 "relations": [{{"source": "...", "target": "...", "type": "..."}}]}}

Rules:
- At most {max_entities} entities and {max_relations} relations.
- Use each entity's name as written in the text.
- Every relation's source and target must be names from your entities list.
- Relation types are short snake_case verbs, e.g. "defines", "applies_to", "part_of", "requires".

Text:
{chunk_text}"""


def build_extract_llm() -> ChatOllama:
    # format="json" makes Ollama constrain decoding to valid JSON.
    return build_llm(num_ctx=4096, num_predict=512, timeout=120, format="json")


def _as_str(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_extraction(raw: str) -> tuple[list[dict], list[dict]]:
    """Turn the model's reply into (entities, relations), dropping anything
    malformed instead of failing: one bad chunk shouldn't sink an ingest."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # tolerate prose or code fences around the object
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return [], []
        try:
            data = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return [], []
    if not isinstance(data, dict):
        return [], []

    entities = []
    for item in data.get("entities") or []:
        if isinstance(item, dict) and _as_str(item.get("name")):
            entities.append({"name": _as_str(item["name"]),
                             "type": _as_str(item.get("type"))})
    relations = []
    for item in data.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source, target = _as_str(item.get("source")), _as_str(item.get("target"))
        if source and target:
            relations.append({"source": source, "target": target,
                              "type": _as_str(item.get("type"))})
    return entities[:MAX_ENTITIES], relations[:MAX_RELATIONS]


def extract_chunk_graph(llm: ChatOllama, chunk_index: int, chunk_text: str) -> ChunkGraph:
    try:
        response = llm.invoke(EXTRACT_PROMPT.format(
            max_entities=MAX_ENTITIES, max_relations=MAX_RELATIONS,
            chunk_text=chunk_text))
        entities, relations = parse_extraction(response.content)
    except Exception:
        logger.exception("extract_chunk_graph failed chunk_index=%d", chunk_index)
        entities, relations = [], []
    return ChunkGraph(chunk_index=chunk_index, entities=entities, relations=relations)
