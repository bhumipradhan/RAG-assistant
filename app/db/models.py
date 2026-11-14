from datetime import UTC, date, datetime, time

from sqlalchemy import Date, DateTime, String, Time, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Metadata only. Chunk text lives in the vector store payload."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("content_hash", "chunking_strategy"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(10))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    chunking_strategy: Mapped[str] = mapped_column(String(20))
    chunk_count: Mapped[int]
    char_count: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Booking(Base):
    __tablename__ = "bookings"
    # One interview per slot; the DB enforces it even if two requests race.
    __table_args__ = (UniqueConstraint("booking_date", "booking_time"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255))
    booking_date: Mapped[date] = mapped_column(Date)
    booking_time: Mapped[time] = mapped_column(Time)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
