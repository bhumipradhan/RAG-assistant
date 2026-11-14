from collections.abc import Sequence

import redis as redis_lib

from app.schemas.chat import ChatMessage
from tests.conftest import last_user
from tests.test_ingestion_api import DOC


def rag_handler(system: str, messages: Sequence[ChatMessage]) -> str:
    if system.startswith("Classify"):
        return "BOOKING" if "book" in last_user(messages).lower() else "QUESTION"
    if system.startswith("Rewrite"):
        return "How much does the Pro plan cost per month?"
    return "The Pro plan costs 50 dollars per month [1]."


def upload(env) -> None:  # type: ignore[no-untyped-def]
    r = env.client.post(
        "/api/v1/documents", files={"file": ("plans.txt", DOC.encode(), "text/plain")}
    )
    assert r.status_code == 201


def test_question_returns_answer_and_sources(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(rag_handler)
    upload(env)
    r = env.client.post(
        "/api/v1/chat", json={"session_id": "s1", "message": "How much is the Pro plan?"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "question" and "50 dollars" in body["answer"]
    assert body["sources"] and body["sources"][0]["filename"] == "plans.txt"


def test_followup_is_rewritten_using_history(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(rag_handler)
    upload(env)
    env.client.post("/api/v1/chat", json={"session_id": "s2", "message": "Tell me about plans"})
    env.client.post("/api/v1/chat", json={"session_id": "s2", "message": "and how much is it?"})
    rewrite_calls = [c for c in env.llm.calls if c[0].startswith("Rewrite")]
    assert len(rewrite_calls) == 1  # only the follow-up needed rewriting
    history = env.client.app.dependency_overrides  # noqa: F841
    assert env.redis.llen("chat:s2") == 4  # 2 turns x (user + assistant)


def test_no_documents_gives_graceful_reply(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(rag_handler)
    r = env.client.post("/api/v1/chat", json={"session_id": "s3", "message": "anything?"})
    assert r.status_code == 200 and "couldn't find" in r.json()["answer"]


def test_history_is_capped(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(rag_handler)
    for i in range(8):
        env.client.post("/api/v1/chat", json={"session_id": "s4", "message": f"question {i}"})
    assert env.redis.llen("chat:s4") == 10  # chat_history_max_messages default


def test_redis_down_returns_503(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(rag_handler)

    class Broken:
        def lrange(self, *a: object, **k: object) -> list[str]:
            raise redis_lib.ConnectionError("down")

        def get(self, *a: object, **k: object) -> str:
            raise redis_lib.ConnectionError("down")

    from app.core import dependencies as deps

    env.client.app.dependency_overrides[deps.get_redis] = lambda: Broken()  # type: ignore[attr-defined]
    r = env.client.post("/api/v1/chat", json={"message": "hi"})
    assert r.status_code == 503


def test_invalid_session_id_rejected(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(rag_handler)
    r = env.client.post("/api/v1/chat", json={"session_id": "bad id!", "message": "hi"})
    assert r.status_code == 422
