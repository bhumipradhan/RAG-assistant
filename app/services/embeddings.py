"""Embedding providers behind one small interface."""

from collections.abc import Sequence
from functools import partial
from typing import Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import OpenAI, OpenAIError

from app.core.config import EmbeddingProvider, Settings
from app.core.exceptions import ExternalServiceError
from app.services.llm import is_retryable_gemini_error
from app.services.retry import retry_call

_OPENAI_BATCH = 100
_GEMINI_BATCH = 50


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


class GeminiEmbedder:
    """Gemini embeddings via the same API key as the chat model (free tier available).

    Uses separate task types for documents and queries, which improves retrieval quality.
    """

    def __init__(
        self, api_key: str, model: str, dimension: int, client: genai.Client | None = None
    ) -> None:
        self._client = client or genai.Client(api_key=api_key)
        self._model = model
        self._dimension = dimension

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        config = genai_types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self._dimension
        )
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _GEMINI_BATCH):
            batch = list(texts[i : i + _GEMINI_BATCH])
            try:
                resp = retry_call(
                    partial(
                        self._client.models.embed_content,
                        model=self._model,
                        contents=batch,
                        config=config,
                    ),
                    should_retry=is_retryable_gemini_error,
                )
            except genai_errors.APIError as exc:
                raise ExternalServiceError(
                    f"Embedding provider error ({exc.code}): {exc.message}"
                ) from exc
            for emb in resp.embeddings or []:
                vectors.append(_normalize(emb.values or []))
        if len(vectors) != len(texts):
            raise ExternalServiceError("Embedding provider returned an unexpected result.")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def _normalize(vec: list[float]) -> list[float]:
    """Shortened Gemini embeddings are not unit length; normalize so scores are comparable."""
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider is EmbeddingProvider.GEMINI:
        if settings.gemini_api_key is None:
            raise ExternalServiceError("GEMINI_API_KEY is required for Gemini embeddings.")
        return GeminiEmbedder(
            settings.gemini_api_key.get_secret_value(),
            settings.embedding_model,
            settings.embedding_dimension,
        )
    if settings.embedding_provider is EmbeddingProvider.FASTEMBED:
        return FastEmbedEmbedder(settings.embedding_model)
    if settings.openai_api_key is None:
        raise ExternalServiceError("OPENAI_API_KEY is required for OpenAI embeddings.")
    return OpenAIEmbedder(settings.openai_api_key.get_secret_value(), settings.embedding_model)
