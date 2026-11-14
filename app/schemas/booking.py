from datetime import date, time

from pydantic import BaseModel, ConfigDict


class BookingExtraction(BaseModel):
    """Raw, UNTRUSTED output of the LLM. Everything is optional and unvalidated."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    email: str | None = None
    booking_date: str | None = None
    booking_time: str | None = None
    confirmation: bool | None = None
    cancel: bool = False


class BookingState(BaseModel):
    """Validated, partially-filled booking. Persisted in Redis between turns."""

    name: str | None = None
    email: str | None = None
    booking_date: date | None = None
    booking_time: time | None = None
    awaiting_confirmation: bool = False

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.name:
            missing.append("name")
        if not self.email:
            missing.append("email")
        if self.booking_date is None:
            missing.append("booking_date")
        if self.booking_time is None:
            missing.append("booking_time")
        return missing
