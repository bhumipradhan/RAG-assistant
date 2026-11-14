"""Interview booking: LLM extracts fields, CODE validates them, Redis keeps state between turns.

The LLM is never trusted to decide that a booking is valid or complete.
"""

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time

from pydantic import EmailStr, TypeAdapter, ValidationError

from app.db.repositories import BookingRepository, SlotAlreadyBookedError
from app.schemas.booking import BookingExtraction, BookingState
from app.schemas.chat import BookingOut, ChatMessage, Role
from app.services.llm import LLMClient
from app.services.memory import BookingStateStore
from app.services.prompts import BOOKING_EXTRACT_SYSTEM

logger = logging.getLogger(__name__)

_EMAIL = TypeAdapter(EmailStr)
_FIELD_LABELS = {
    "name": "your full name",
    "email": "your email address",
    "booking_date": "your preferred date",
    "booking_time": "your preferred time",
}


@dataclass(frozen=True)
class BookingTurn:
    reply: str
    booking: BookingOut | None = None


def parse_extraction(raw: str) -> BookingExtraction:
    """Tolerant JSON parsing: strips code fences / chatter. Falls back to 'nothing extracted'."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return BookingExtraction.model_validate(json.loads(match.group(0)))
        except (json.JSONDecodeError, ValidationError):
            pass
    logger.warning("Could not parse booking extraction output: %r", raw[:200])
    return BookingExtraction()


class BookingService:
    def __init__(
        self,
        *,
        llm: LLMClient,
        state_store: BookingStateStore,
        repo: BookingRepository,
        now: Callable[[], datetime],
    ) -> None:
        self._llm = llm
        self._states = state_store
        self._repo = repo
        self._now = now

    def is_in_progress(self, session_id: str) -> bool:
        return self._states.get(session_id) is not None

    def handle(self, session_id: str, message: str, history: Sequence[ChatMessage]) -> BookingTurn:
        state = self._states.get(session_id) or BookingState()
        extraction = self._extract(message, history, state)

        if extraction.cancel:
            self._states.clear(session_id)
            return BookingTurn("No problem, I've cancelled the booking request.")

        state, problems, changed = self._merge(state, extraction)

        if state.awaiting_confirmation and not changed and not problems:
            if extraction.confirmation is True:
                return self._persist(session_id, state)
            if extraction.confirmation is False:
                state.awaiting_confirmation = False
                self._states.save(session_id, state)
                return BookingTurn("Okay, what would you like to change?")

        missing = state.missing_fields()
        if missing:
            state.awaiting_confirmation = False
            self._states.save(session_id, state)
            return BookingTurn(self._ask_for(missing, problems))

        state.awaiting_confirmation = True
        self._states.save(session_id, state)
        return BookingTurn(self._summary(state, problems))

    # ---- LLM extraction -------------------------------------------------------------------
    def _extract(
        self, message: str, history: Sequence[ChatMessage], state: BookingState
    ) -> BookingExtraction:
        today = self._now()
        system = BOOKING_EXTRACT_SYSTEM.format(
            today=today.date().isoformat(),
            weekday=today.strftime("%A"),
            known=state.model_dump_json(exclude={"awaiting_confirmation"}),
        )
        messages = [*history[-6:], ChatMessage(role=Role.USER, content=message)]
        return parse_extraction(self._llm.complete(system, messages, max_tokens=300))

    # ---- Validation in code ---------------------------------------------------------------
    def _merge(
        self, state: BookingState, ex: BookingExtraction
    ) -> tuple[BookingState, list[str], bool]:
        new = state.model_copy()
        problems: list[str] = []
        changed = False
        now = self._now()

        if ex.name is not None and ex.name.strip():
            name = ex.name.strip()
            if 2 <= len(name) <= 100:
                changed |= name != new.name
                new.name = name
            else:
                problems.append("That name doesn't look right.")

        if ex.email is not None and ex.email.strip():
            try:
                email = str(_EMAIL.validate_python(ex.email.strip()))
                changed |= email != new.email
                new.email = email
            except ValidationError:
                problems.append(f"'{ex.email}' isn't a valid email address.")

        if ex.booking_date is not None:
            try:
                d = date.fromisoformat(ex.booking_date.strip())
                if d < now.date():
                    problems.append("That date is in the past.")
                else:
                    changed |= d != new.booking_date
                    new.booking_date = d
            except ValueError:
                problems.append("I couldn't understand that date.")

        if ex.booking_time is not None:
            try:
                t = time.fromisoformat(ex.booking_time.strip()).replace(second=0, microsecond=0)
                changed |= t != new.booking_time
                new.booking_time = t
            except ValueError:
                problems.append("I couldn't understand that time.")

        # Cross-field rule: a slot earlier today is already gone.
        if (
            new.booking_date == now.date()
            and new.booking_time is not None
            and new.booking_time <= now.time()
        ):
            problems.append("That time has already passed today.")
            new.booking_time = None

        if changed:
            new.awaiting_confirmation = False
        return new, problems, changed

    # ---- Persistence ----------------------------------------------------------------------
    def _persist(self, session_id: str, state: BookingState) -> BookingTurn:
        assert state.name and state.email and state.booking_date and state.booking_time
        try:
            record = self._repo.add(
                session_id=session_id,
                name=state.name,
                email=state.email,
                booking_date=state.booking_date,
                booking_time=state.booking_time,
            )
        except SlotAlreadyBookedError:
            state.booking_time = None
            state.awaiting_confirmation = False
            self._states.save(session_id, state)
            return BookingTurn("Sorry, that slot was just taken. Could you pick another time?")
        self._states.clear(session_id)
        out = BookingOut.model_validate(record)
        return BookingTurn(
            f"You're all set, {out.name}! Interview booked for "
            f"{out.booking_date:%A, %d %B %Y} at {out.booking_time:%H:%M}. "
            f"A confirmation will go to {out.email}.",
            out,
        )

    # ---- Reply text (deterministic templates, easy to test) -------------------------------
    @staticmethod
    def _ask_for(missing: list[str], problems: list[str]) -> str:
        labels = [_FIELD_LABELS[f] for f in missing]
        listed = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + f" and {labels[-1]}"
        return " ".join([*problems, f"To book your interview, could you share {listed}?"])

    @staticmethod
    def _summary(state: BookingState, problems: list[str]) -> str:
        assert state.booking_date and state.booking_time
        details = (
            f"Name: {state.name}, Email: {state.email}, "
            f"Date: {state.booking_date:%A, %d %B %Y}, Time: {state.booking_time:%H:%M}."
        )
        return " ".join(
            [*problems, f"Please confirm your booking. {details} Is this correct (yes/no)?"]
        )
