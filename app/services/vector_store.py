"""Vector database access. The rest of the app only knows the `VectorStore` Protocol."""

import uuid
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.exceptions import ExternalServiceError

_UPSERT_BATCH = 128
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class SearchHit(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class VectorStore(Protocol):
    def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]: ...

    def delete_document(self, document_id: str) -> None: ...


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension
        self._ready = False

    def _ensure_collection(self) -> None:
        if self._ready:
            return
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            )
            self._client.create_payload_index(
                self._collection, "document_id", PayloadSchemaType.KEYWORD
            )
        self._ready = True

    def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            PointStruct(
                # Deterministic id => re-ingesting the same doc overwrites instead of duplicating.
                id=str(uuid.uuid5(_NAMESPACE, f"{document_id}:{i}")),
                vector=list(vec),
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "text": text,
                },
            )
            for i, (text, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]
        try:
            self._ensure_collection()
            for i in range(0, len(points), _UPSERT_BATCH):
                self._client.upsert(self._collection, points=points[i : i + _UPSERT_BATCH])
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise ExternalServiceError(f"Vector store error: {exc}") from exc

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        query_filter = (
            Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=list(document_ids)))])
            if document_ids
            else None
        )
        try:
            self._ensure_collection()
            result = self._client.query_points(
                self._collection,
                query=list(vector),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise ExternalServiceError(f"Vector store error: {exc}") from exc
        hits: list[SearchHit] = []
        for point in result.points:
            p = point.payload or {}
            hits.append(
                SearchHit(
                    document_id=str(p.get("document_id", "")),
                    filename=str(p.get("filename", "")),
                    chunk_index=int(p.get("chunk_index", 0)),
                    text=str(p.get("text", "")),
                    score=point.score,
                )
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        try:
            self._ensure_collection()
            self._client.delete(
                self._collection,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(key="document_id", match=MatchValue(value=document_id))
                        ]
                    )
                ),
            )
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            raise ExternalServiceError(f"Vector store error: {exc}") from exc
