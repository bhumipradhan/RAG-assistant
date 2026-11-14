"""Embedding providers behind one small interface."""

from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI, OpenAIError

from app.core.config import EmbeddingProvider, Settings
from app.core.exceptions import ExternalServiceError

_OPENAI_BATCH = 100


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            for i in range(0, len(texts), _OPENAI_BATCH):
                batch = list(texts[i : i + _OPENAI_BATCH])
                resp = self._client.embeddings.create(model=self._model, input=batch)
                vectors.extend(item.embedding for item in resp.data)
        except OpenAIError as exc:
            raise ExternalServiceError(f"Embedding provider error: {exc}") from exc
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    """Runs a small model locally. No API key, no network after the first model download."""

    def __init__(self, model: str) -> None:
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self._model = TextEmbedding(model_name=model)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider is EmbeddingProvider.FASTEMBED:
        return FastEmbedEmbedder(settings.embedding_model)
    if settings.openai_api_key is None:
        raise ExternalServiceError("OPENAI_API_KEY is required for OpenAI embeddings.")
    return OpenAIEmbedder(settings.openai_api_key.get_secret_value(), settings.embedding_model)
