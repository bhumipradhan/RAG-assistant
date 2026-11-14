import hashlib
import re
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime

import fakeredis
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy.orm import Session, sessionmaker

from app.core import dependencies as deps
from app.core.config import Settings, get_settings
from app.db.session import create_session_factory
from app.main import create_app
from app.schemas.chat import ChatMessage, Role
from app.services.vector_store import QdrantVectorStore

DIM = 128
FIXED_NOW = datetime(2026, 11, 14, 10, 0, tzinfo=UTC)  # a Saturday


class FakeEmbedder:
    """Bag-of-words hashing: texts sharing words get similar vectors. Deterministic, offline."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM:
    """Scripted LLM. `handler(system, messages)` decides the reply; calls are recorded."""

    def __init__(self, handler: Callable[[str, Sequence[ChatMessage]], str]) -> None:
        self.handler = handler
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    def complete(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append((system, list(messages)))
        return self.handler(system, messages)


class Env:
    def __init__(self, client: TestClient, llm: FakeLLM, redis_client: Redis) -> None:
        self.client = client
        self.llm = llm
        self.redis = redis_client


@pytest.fixture
def make_env() -> Iterator[Callable[[Callable[[str, Sequence[ChatMessage]], str]], Env]]:
    created: list[TestClient] = []

    def _make(handler: Callable[[str, Sequence[ChatMessage]], str]) -> Env:
        factory: sessionmaker[Session] = create_session_factory("sqlite:///:memory:")
        store = QdrantVectorStore(QdrantClient(":memory:"), "test", DIM)
        llm = FakeLLM(handler)
        redis_client: Redis = fakeredis.FakeRedis(decode_responses=True)
        settings = Settings(retrieval_top_k=2, chunk_size=200, chunk_overlap=20)

        def _db() -> Iterator[Session]:
            with factory() as s:
                yield s

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[deps.get_db] = _db
        app.dependency_overrides[deps.get_embedder] = lambda: FakeEmbedder()
        app.dependency_overrides[deps.get_vector_store] = lambda: store
        app.dependency_overrides[deps.get_llm] = lambda: llm
        app.dependency_overrides[deps.get_redis] = lambda: redis_client
        client = TestClient(app)
        created.append(client)
        return Env(client, llm, redis_client)

    yield _make
    for c in created:
        c.close()


def last_user(messages: Sequence[ChatMessage]) -> str:
    return next(m.content for m in reversed(messages) if m.role is Role.USER)
