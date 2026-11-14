"""Ingestion pipeline: bytes -> text -> chunks -> embeddings -> vector store + SQL metadata."""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.core.exceptions import EmptyDocumentError, FileTooLargeError
from app.db.models import Document
from app.db.repositories import DocumentRepository
from app.schemas.ingestion import ChunkingStrategy
from app.services.chunking.registry import get_chunker
from app.services.embeddings import Embedder
from app.services.extraction import extract_text
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    duplicate: bool


class IngestionService:
    def __init__(
        self,
        *,
        repo: DocumentRepository,
        embedder: Embedder,
        store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
        max_upload_bytes: int,
    ) -> None:
        self._repo = repo
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_bytes = max_upload_bytes

    def ingest(
        self, filename: str, stream: BinaryIO, strategy: ChunkingStrategy
    ) -> IngestionResult:
        data = stream.read(self._max_bytes + 1)  # never read more than the limit + 1 byte
        if len(data) > self._max_bytes:
            raise FileTooLargeError(
                f"File exceeds the {self._max_bytes // (1024 * 1024)} MB limit."
            )

        safe_name = Path(filename).name[:255] or "unnamed"
        text, file_type = extract_text(safe_name, data)

        content_hash = hashlib.sha256(data).hexdigest()
        existing = self._repo.get_by_hash(content_hash, strategy.value)
        if existing is not None:  # idempotency: same file + same strategy => no new vectors
            return IngestionResult(existing, duplicate=True)

        chunks = get_chunker(strategy, self._chunk_size, self._chunk_overlap).chunk(text)
        if not chunks:
            raise EmptyDocumentError("The document produced no chunks.")

        vectors = self._embedder.embed_documents(chunks)
        document_id = str(uuid.uuid4())
        self._store.upsert_chunks(document_id, safe_name, chunks, vectors)
        try:
            document = self._repo.add(
                Document(
                    id=document_id,
                    filename=safe_name,
                    file_type=file_type.value,
                    content_hash=content_hash,
                    chunking_strategy=strategy.value,
                    chunk_count=len(chunks),
                    char_count=len(text),
                )
            )
        except Exception:
            # Compensate: don't leave orphan vectors behind if the SQL write fails.
            logger.exception("Metadata save failed; removing vectors for %s", document_id)
            self._store.delete_document(document_id)
            raise
        return IngestionResult(document, duplicate=False)
