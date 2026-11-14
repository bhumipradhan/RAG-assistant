"""Central, typed configuration. Every setting can be overridden by an env var / .env file."""

from enum import StrEnum
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class EmbeddingProvider(StrEnum):
    OPENAI = "openai"
    FASTEMBED = "fastembed"
    GEMINI = "gemini"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Assistant"

    # LLM
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 30  # per request to the LLM / embedding provider (Gemini)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # Embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Infrastructure
    database_url: str = "sqlite:///./app.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    redis_url: str = "redis://localhost:6379/0"

    # Behaviour
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_upload_mb: int = 10
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.0
    chat_history_max_messages: int = 10
    chat_ttl_seconds: int = 3600
    timezone: str = "UTC"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
