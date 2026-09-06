from langchain_qdrant import QdrantVectorStore
from app.utils.qdrant import client
from app.utils.embedding import embedding_model

vector_store = QdrantVectorStore(
    client=client, collection_name="documents", embedding=embedding_model
)
