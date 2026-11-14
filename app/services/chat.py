"""Chat orchestration: memory -> intent -> (booking | RAG) -> memory."""

import uuid

from app.schemas.chat import ChatMessage, ChatResponse, Intent, Role
from app.services.booking import BookingService
from app.services.intent import IntentClassifier
from app.services.memory import ChatMemory
from app.services.rag import RagService


class ChatService:
    def __init__(
        self,
        *,
        memory: ChatMemory,
        intent: IntentClassifier,
        rag: RagService,
        booking: BookingService,
    ) -> None:
        self._memory = memory
        self._intent = intent
        self._rag = rag
        self._booking = booking

    def chat(self, session_id: str | None, message: str) -> ChatResponse:
        sid = session_id or uuid.uuid4().hex
        history = self._memory.get_history(sid)
        in_progress = self._booking.is_in_progress(sid)
        intent = self._intent.classify(message, history, in_progress)

        if intent is Intent.BOOKING:
            turn = self._booking.handle(sid, message, history)
            response = ChatResponse(
                session_id=sid, answer=turn.reply, intent=intent, booking=turn.booking
            )
        else:
            result = self._rag.answer(message, history)
            response = ChatResponse(
                session_id=sid, answer=result.answer, intent=intent, sources=result.sources
            )

        self._memory.append(
            sid,
            ChatMessage(role=Role.USER, content=message),
            ChatMessage(role=Role.ASSISTANT, content=response.answer),
        )
        return response
