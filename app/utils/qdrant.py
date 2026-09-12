from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from app.core.config import settings

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

COLLECTION_NAME = "documents"
COLLECTION_NAME_v2 = "documents_v2"

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="metadata.doc_id",
    field_schema=PayloadSchemaType.KEYWORD,
)

if not client.collection_exists(COLLECTION_NAME_v2):
    client.create_collection(
        collection_name=COLLECTION_NAME_v2,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

client.create_payload_index(
    collection_name=COLLECTION_NAME_v2,
    field_name="metadata.doc_id",
    field_schema=PayloadSchemaType.KEYWORD,
)
