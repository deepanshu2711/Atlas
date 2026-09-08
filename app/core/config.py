from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./database.db"
    qdrant_url: str
    qdrant_api_key: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
