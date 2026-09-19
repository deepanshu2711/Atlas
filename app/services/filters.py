from qdrant_client.models import FieldCondition, Filter, MatchValue


def doc_filter(doc_id: str) -> Filter:
    return Filter(must=[
        FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))
    ])
