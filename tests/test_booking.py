import json
from collections.abc import Sequence

from sqlalchemy import select

from app.core import dependencies as deps
from app.db.models import Booking
from app.schemas.chat import ChatMessage
from app.services.booking import parse_extraction
from tests.conftest import FIXED_NOW, last_user  # noqa: F401


def scripted(extractions: list[dict[str, object]]):  # type: ignore[no-untyped-def]
    """Intent is always BOOKING; extraction replies are consumed in order."""
    queue = list(extractions)

    def handler(system: str, messages: Sequence[ChatMessage]) -> str:
        if system.startswith("Classify"):
            return "BOOKING"
        return json.dumps(queue.pop(0))

    return handler


def say(env, text: str) -> dict[str, object]:  # type: ignore[no-untyped-def]
    r = env.client.post("/api/v1/chat", json={"session_id": "b1", "message": text})
    assert r.status_code == 200
    return r.json()  # type: ignore[no-any-return]


def test_full_multi_turn_booking(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(
        scripted(
            [
                {"name": None},
                {"name": "Jane Doe", "email": "jane@example.org"},
                {"booking_date": "2099-01-05", "booking_time": "14:30"},
                {"confirmation": True},
            ]
        )
    )
    first = say(env, "I'd like to book an interview")
    assert "full name" in str(first["answer"])

    second = say(env, "I'm Jane Doe, jane@example.org")
    assert "preferred date" in str(second["answer"]) and "full name" not in str(second["answer"])

    third = say(env, "5 Jan 2099 at 2:30pm")
    assert "Is this correct" in str(third["answer"])

    done = say(env, "yes")
    assert done["booking"] is not None and "all set" in str(done["answer"])

    db = next(env.client.app.dependency_overrides[deps.get_db]())  # type: ignore[attr-defined]
    rows = db.scalars(select(Booking)).all()
    assert len(rows) == 1 and rows[0].email == "jane@example.org"
    assert env.redis.get("booking:b1") is None  # state cleared


def test_invalid_email_and_past_date_rejected_in_code(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(
        scripted([{"name": "Jane Doe", "email": "not-an-email", "booking_date": "2001-01-01"}])
    )
    reply = str(say(env, "book: Jane Doe, not-an-email, 2001-01-01")["answer"])
    assert "isn't a valid email" in reply
    assert "in the past" in reply


def test_cancel_clears_state(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(scripted([{"name": "Jane Doe"}, {"cancel": True}]))
    say(env, "book an interview, I'm Jane Doe")
    assert env.redis.get("booking:b1") is not None
    say(env, "never mind")
    assert env.redis.get("booking:b1") is None


def test_change_after_summary_requires_reconfirmation(make_env) -> None:  # type: ignore[no-untyped-def]
    full = {
        "name": "Jane Doe",
        "email": "jane@example.org",
        "booking_date": "2099-01-05",
        "booking_time": "14:30",
    }
    env = make_env(scripted([full, {"booking_time": "16:00", "confirmation": True}]))
    say(env, "book it all")
    again = say(env, "actually make it 4pm, yes")
    assert "Is this correct" in str(again["answer"])  # changed => asks again, no booking yet
    assert again["booking"] is None


def test_garbage_llm_output_is_safe() -> None:
    ex = parse_extraction("Sure! Here you go: not json at all")
    assert ex.name is None and ex.cancel is False


def test_fenced_json_is_parsed() -> None:
    ex = parse_extraction('```json\n{"name": "Al", "extra": 1}\n```')
    assert ex.name == "Al"
