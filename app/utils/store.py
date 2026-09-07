from langchain_qdrant import QdrantVectorStore
from app.utils.qdrant import client, COLLECTION_NAME
from app.utils.embedding import embedding_model

vector_store = QdrantVectorStore(
    client=client, collection_name=COLLECTION_NAME, embedding=embedding_model
)
