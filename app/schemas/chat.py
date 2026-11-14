from datetime import date, time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


class Intent(StrEnum):
    QUESTION = "question"
    BOOKING = "booking"


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )
    message: str = Field(min_length=1, max_length=2000)


class SourceOut(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    snippet: str


class BookingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    email: str
    booking_date: date
    booking_time: time


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: Intent
    sources: list[SourceOut] = Field(default_factory=list)
    booking: BookingOut | None = None
