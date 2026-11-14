from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


class FileType(StrEnum):
    PDF = "pdf"
    TXT = "txt"


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    chunking_strategy: str
    chunk_count: int
    char_count: int
    created_at: datetime
    duplicate: bool = False
