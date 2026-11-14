"""Redis-backed conversation memory and booking state, keyed by session id."""

from collections.abc import Iterator
from contextlib import contextmanager

from redis import Redis
from redis.exceptions import RedisError

from app.core.exceptions import ExternalServiceError
from app.schemas.booking import BookingState
from app.schemas.chat import ChatMessage


@contextmanager
def _redis_guard() -> Iterator[None]:
    """Translate low-level Redis failures into one domain error (-> HTTP 503)."""
    try:
        yield
    except RedisError as exc:
        raise ExternalServiceError("Conversation memory (Redis) is unavailable.") from exc


class ChatMemory:
    """Chat history as a capped Redis list. Oldest messages fall off the front."""

    def __init__(self, client: Redis, max_messages: int, ttl_seconds: int) -> None:
        self._client = client
        self._max = max_messages
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:{session_id}"

    def get_history(self, session_id: str) -> list[ChatMessage]:
        with _redis_guard():
            raw = self._client.lrange(self._key(session_id), 0, -1)
        return [ChatMessage.model_validate_json(item) for item in raw]

    def append(self, session_id: str, *messages: ChatMessage) -> None:
        key = self._key(session_id)
        with _redis_guard():
            pipe = self._client.pipeline()
            pipe.rpush(key, *(m.model_dump_json() for m in messages))
            pipe.ltrim(key, -self._max, -1)
            pipe.expire(key, self._ttl)
            pipe.execute()


class BookingStateStore:
    """In-progress booking, stored separately from chat history so it can't be truncated away."""

    def __init__(self, client: Redis, ttl_seconds: int) -> None:
        self._client = client
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"booking:{session_id}"

    def get(self, session_id: str) -> BookingState | None:
        with _redis_guard():
            raw = self._client.get(self._key(session_id))
        return BookingState.model_validate_json(raw) if raw else None

    def save(self, session_id: str, state: BookingState) -> None:
        with _redis_guard():
            self._client.set(self._key(session_id), state.model_dump_json(), ex=self._ttl)

    def clear(self, session_id: str) -> None:
        with _redis_guard():
            self._client.delete(self._key(session_id))
