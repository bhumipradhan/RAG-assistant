from collections.abc import Sequence

from app.schemas.chat import ChatMessage, Intent, Role
from app.services.llm import LLMClient
from app.services.prompts import INTENT_SYSTEM

_BOOKING_NOTE = (
    "A booking is currently IN PROGRESS: replies that give details, confirm, or cancel it "
    "are BOOKING, even if short (e.g. 'yes', 'tomorrow at 3pm')."
)


class IntentClassifier:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def classify(
        self, message: str, history: Sequence[ChatMessage], booking_in_progress: bool
    ) -> Intent:
        system = INTENT_SYSTEM.format(booking_note=_BOOKING_NOTE if booking_in_progress else "")
        messages = [*history[-4:], ChatMessage(role=Role.USER, content=message)]
        label = self._llm.complete(system, messages, max_tokens=5).strip().upper()
        return Intent.BOOKING if label.startswith("BOOKING") else Intent.QUESTION
