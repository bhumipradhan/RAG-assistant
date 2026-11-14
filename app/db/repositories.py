import uuid
from datetime import date, time

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Booking, Document


class SlotAlreadyBookedError(Exception):
    pass


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_hash(self, content_hash: str, strategy: str) -> Document | None:
        stmt = select(Document).where(
            Document.content_hash == content_hash, Document.chunking_strategy == strategy
        )
        return self._db.scalars(stmt).first()

    def add(self, document: Document) -> Document:
        self._db.add(document)
        self._db.commit()
        return document

    def list_all(self) -> list[Document]:
        return list(self._db.scalars(select(Document).order_by(Document.created_at.desc())))


class BookingRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(
        self, *, session_id: str, name: str, email: str, booking_date: date, booking_time: time
    ) -> Booking:
        booking = Booking(
            id=str(uuid.uuid4()),
            session_id=session_id,
            name=name,
            email=email,
            booking_date=booking_date,
            booking_time=booking_time,
        )
        self._db.add(booking)
        try:
            self._db.commit()
        except IntegrityError as exc:
            self._db.rollback()
            raise SlotAlreadyBookedError from exc
        return booking
