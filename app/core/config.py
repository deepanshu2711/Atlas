from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./database.db"
    qdrant_url: str
    qdrant_api_key: str
    docling_ocr_enabled: bool = False
    docling_table_structure_enabled: bool = True
    qdrant_timeout_seconds: int = 180
    retrieval_mode: Literal["dense", "bm25",
                            "hybrid", "graph", "hybrid_graph"] = "dense"
    final_k: int = 8
    candidate_k: int = 40
    rrf_k: int = 60
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    contextual_chunks_enabled: bool = False
    graph_enabled: bool = False
    graph_k: int = 20
    kuzu_path: str = "data/graph.kuzu"
    agentic_enabled: bool = True
    agent_max_hops: int = 4
    agent_step_k: int = 3
    agent_num_ctx: int = 16384
    verify_enabled: bool = True
    verify_max_claims: int = 8
    verify_repair_attempts: int = 1
    # Ollama model that grades eval answers; empty means the answering model.
    judge_model: str = ""

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
