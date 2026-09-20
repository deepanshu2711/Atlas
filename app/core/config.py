from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./database.db"
    qdrant_url: str
    qdrant_api_key: str
    docling_ocr_enabled: bool = False
    docling_table_structure_enabled: bool = False
    qdrant_timeout_seconds: int = 180
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "dense"
    final_k: int = 8
    candidate_k: int = 40
    rrf_k: int = 60
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
