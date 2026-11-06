from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """All application settings, loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        extra="ignore",
    )

    app_name: str = "hospital-rag"
    environment: str = "dev"
    log_level: str = "INFO"

    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 250
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "hospital_chunks"

    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    min_score: float = 0.15
    top_k: int = 3    


def get_settings() -> Settings:
    return Settings()