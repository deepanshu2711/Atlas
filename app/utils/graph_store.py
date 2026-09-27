import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import kuzu

from app.core.config import settings

# Two node types and two edge types:
#   (Chunk)-[:MENTIONS]->(Entity)          which chunks talk about an entity
#   (Entity)-[:RELATED {type}]->(Entity)   what the LLM said links two entities
# Entity ids are scoped to a document ("<doc_id>:<normalised name>") because
# retrieval is single-document: the same name in two PDFs is two nodes.
_SCHEMA = [
    "CREATE NODE TABLE IF NOT EXISTS Chunk("
    "id STRING, doc_id STRING, chunk_index INT64, PRIMARY KEY (id))",
    "CREATE NODE TABLE IF NOT EXISTS Entity("
    "id STRING, doc_id STRING, name STRING, type STRING, PRIMARY KEY (id))",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Chunk TO Entity)",
    "CREATE REL TABLE IF NOT EXISTS RELATED(FROM Entity TO Entity, type STRING)",
]

_WORD_RE = re.compile(r"\w+")
# Entity names shorter than this ("AI", "EU") match inside far too many
# questions to say anything about which chunk is relevant.
MIN_ENTITY_CHARS = 3


def normalize(name: str) -> str:
    """Lowercase and collapse punctuation/whitespace, so "the AI Act" and
    "The  AI-Act" become the same node."""
    return " ".join(_WORD_RE.findall(name.lower()))


@dataclass
class ChunkGraph:
    """What the extractor found in one chunk."""
    chunk_index: int
    entities: list[dict] = field(default_factory=list)   # {"name", "type"}
    relations: list[dict] = field(default_factory=list)  # {"source", "target", "type"}


class GraphStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)
        # Ingestion runs in Starlette's threadpool while queries run
        # alongside it; one lock keeps a single statement/transaction on the
        # connection at a time.
        self._lock = threading.Lock()
        for statement in _SCHEMA:
            self._conn.execute(statement)

    def _rows(self, query: str, params: dict | None = None) -> list[list]:
        with self._lock:
            return self._conn.execute(query, parameters=params or {}).get_all()

    @staticmethod
    def _entity_id(doc_id: str, name: str) -> str:
        return f"{doc_id}:{normalize(name)}"

    def has_document(self, doc_id: str) -> bool:
        rows = self._rows(
            "MATCH (c:Chunk) WHERE c.doc_id = $doc_id RETURN count(*)",
            {"doc_id": doc_id})
        return rows[0][0] > 0

    def replace_document(self, doc_id: str, chunks: list[ChunkGraph]) -> int:
        """Delete the document's old graph and write the new one in a single
        transaction, so a query never sees half of a re-ingested document.
        Returns the number of distinct entities written."""
        entity_ids = set()
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN TRANSACTION")
            try:
                for table in ("Chunk", "Entity"):
                    # DETACH DELETE also drops the edges touching each node
                    conn.execute(
                        f"MATCH (n:{table}) WHERE n.doc_id = $doc_id DETACH DELETE n",
                        {"doc_id": doc_id})

                for chunk in chunks:
                    chunk_id = f"{doc_id}:{chunk.chunk_index}"
                    conn.execute(
                        "CREATE (:Chunk {id: $id, doc_id: $doc_id, chunk_index: $idx})",
                        {"id": chunk_id, "doc_id": doc_id, "idx": chunk.chunk_index})

                    for entity in chunk.entities:
                        if not normalize(entity["name"]):
                            continue
                        entity_id = self._merge_entity(
                            conn, doc_id, entity["name"], entity.get("type", ""))
                        entity_ids.add(entity_id)
                        conn.execute(
                            "MATCH (c:Chunk {id: $cid}), (e:Entity {id: $eid}) "
                            "MERGE (c)-[:MENTIONS]->(e)",
                            {"cid": chunk_id, "eid": entity_id})

                    for rel in chunk.relations:
                        if not (normalize(rel["source"]) and normalize(rel["target"])):
                            continue
                        # the model sometimes names an endpoint it didn't list
                        # as an entity; create it rather than drop the edge
                        src = self._merge_entity(conn, doc_id, rel["source"], "")
                        dst = self._merge_entity(conn, doc_id, rel["target"], "")
                        if src == dst:
                            continue
                        entity_ids.update((src, dst))
                        conn.execute(
                            "MATCH (a:Entity {id: $src}), (b:Entity {id: $dst}) "
                            "MERGE (a)-[:RELATED {type: $type}]->(b)",
                            {"src": src, "dst": dst, "type": rel.get("type", "")})
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return len(entity_ids)

    def _merge_entity(self, conn, doc_id: str, name: str, type_: str) -> str:
        entity_id = self._entity_id(doc_id, name)
        # MERGE = create if missing, else reuse: an entity named in ten
        # chunks is one node with ten MENTIONS edges.
        conn.execute(
            "MERGE (e:Entity {id: $id}) "
            "ON CREATE SET e.doc_id = $doc_id, e.name = $name, e.type = $type",
            {"id": entity_id, "doc_id": doc_id, "name": normalize(name), "type": type_})
        return entity_id

    def match_entities(self, doc_id: str, query: str) -> list[str]:
        """Ids of the document's entities whose full name appears, as whole
        words, in the query. No LLM call, so graph search adds no model latency."""
        padded_query = f" {normalize(query)} "
        rows = self._rows(
            "MATCH (e:Entity) WHERE e.doc_id = $doc_id RETURN e.id, e.name",
            {"doc_id": doc_id})
        return [entity_id for entity_id, name in rows
                if len(name) >= MIN_ENTITY_CHARS and f" {name} " in padded_query]

    def chunk_scores(self, entity_ids: list[str], neighbour_weight: float = 0.5) -> dict[int, float]:
        """Score chunks by the matched entities they mention (1 point each)
        plus the entities one RELATED hop away that they mention
        (`neighbour_weight` each). The hop is what lets a question about X
        reach the chunk about Y when the graph says X relates to Y."""
        if not entity_ids:
            return {}
        params = {"ids": entity_ids}
        scores: dict[int, float] = {}
        direct = self._rows(
            "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $ids "
            "RETURN c.chunk_index, count(DISTINCT e)", params)
        for chunk_index, n in direct:
            scores[chunk_index] = scores.get(chunk_index, 0.0) + n
        neighbours = self._rows(
            "MATCH (e:Entity)-[:RELATED]-(n:Entity)<-[:MENTIONS]-(c:Chunk) "
            "WHERE e.id IN $ids AND NOT (n.id IN $ids) "
            "RETURN c.chunk_index, count(DISTINCT n)", params)
        for chunk_index, n in neighbours:
            scores[chunk_index] = scores.get(chunk_index, 0.0) + neighbour_weight * n
        return scores


@lru_cache(maxsize=1)
def get_graph_store() -> GraphStore:
    """Opened on first use: Kuzu takes a file lock on the database, so
    nothing should hold it while the graph is off."""
    return GraphStore(settings.kuzu_path)
