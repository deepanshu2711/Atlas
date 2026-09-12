from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./database.db"
    qdrant_url: str
    qdrant_api_key: str
    docling_ocr_enabled: bool = False
    docling_table_structure_enabled: bool = False
    qdrant_timeout_seconds: int = 180

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
