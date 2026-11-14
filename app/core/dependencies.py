"""Dependency wiring. Heavy clients are created once (lru_cache); services are built per request.

Tests replace any of these with `app.dependency_overrides`.
"""

from collections.abc import Iterator
from datetime import datetime
from functools import lru_cache
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import Depends
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.repositories import BookingRepository, DocumentRepository
from app.db.session import create_session_factory
from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.embeddings import Embedder, build_embedder
from app.services.ingestion import IngestionService
from app.services.intent import IntentClassifier
from app.services.llm import LLMClient, build_llm
from app.services.memory import BookingStateStore, ChatMemory
from app.services.rag import RagService
from app.services.vector_store import QdrantVectorStore, VectorStore

SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return create_session_factory(get_settings().database_url)


def get_db() -> Iterator[Session]:
    with get_session_factory()() as session:
        yield session


@lru_cache
def get_embedder() -> Embedder:
    return build_embedder(get_settings())


@lru_cache
def get_llm() -> LLMClient:
    return build_llm(get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    return QdrantVectorStore(
        QdrantClient(url=s.qdrant_url), s.qdrant_collection, s.embedding_dimension
    )


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


DbDep = Annotated[Session, Depends(get_db)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
StoreDep = Annotated[VectorStore, Depends(get_vector_store)]
RedisDep = Annotated[Redis, Depends(get_redis)]


def get_ingestion_service(
    db: DbDep, embedder: EmbedderDep, store: StoreDep, settings: SettingsDep
) -> IngestionService:
    return IngestionService(
        repo=DocumentRepository(db),
        embedder=embedder,
        store=store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
    )


def get_chat_service(
    db: DbDep,
    llm: LLMDep,
    embedder: EmbedderDep,
    store: StoreDep,
    redis_client: RedisDep,
    settings: SettingsDep,
) -> ChatService:
    tz = ZoneInfo(settings.timezone)
    return ChatService(
        memory=ChatMemory(
            redis_client, settings.chat_history_max_messages, settings.chat_ttl_seconds
        ),
        intent=IntentClassifier(llm),
        rag=RagService(
            llm=llm,
            embedder=embedder,
            store=store,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
        ),
        booking=BookingService(
            llm=llm,
            state_store=BookingStateStore(redis_client, settings.chat_ttl_seconds),
            repo=BookingRepository(db),
            now=lambda: datetime.now(tz),
        ),
    )
