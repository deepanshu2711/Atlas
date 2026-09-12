from langchain_qdrant import QdrantVectorStore
from app.utils.qdrant import client, COLLECTION_NAME, COLLECTION_NAME_v2
from app.utils.embedding import embedding_model

vector_store = QdrantVectorStore(
    client=client, collection_name=COLLECTION_NAME, embedding=embedding_model
)

vector_store_v2 = QdrantVectorStore(
    client=client, collection_name=COLLECTION_NAME_v2, embedding=embedding_model
)
