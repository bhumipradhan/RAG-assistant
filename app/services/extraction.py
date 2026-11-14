"""Turn uploaded bytes into clean text. Pure functions, no I/O beyond the bytes given."""

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from app.schemas.ingestion import FileType

_PDF_MAGIC = b"%PDF"


def detect_file_type(filename: str) -> FileType:
    suffix = Path(filename).suffix.lower().lstrip(".")
    try:
        return FileType(suffix)
    except ValueError as exc:
        raise UnsupportedFileTypeError("Only .pdf and .txt files are supported.") from exc

def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"(?:\n[ \t]*){3,}", "\n\n", text)  # 3+ newlines = paragraph break
    text = re.sub(r"\n[ \t]+\n", " ", text)           # "word \n \n word" = just a space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def _extract_pdf(data: bytes) -> str:
    # Validate content, not just the extension.
    if not data.startswith(_PDF_MAGIC):
        raise UnsupportedFileTypeError("File has a .pdf extension but is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except PyPdfError as exc:
        raise UnsupportedFileTypeError("The PDF could not be read (corrupt or encrypted).") from exc


def extract_text(filename: str, data: bytes) -> tuple[str, FileType]:
    file_type = detect_file_type(filename)
    raw = _extract_pdf(data) if file_type is FileType.PDF else data.decode("utf-8", "replace")
    text = _normalize(raw)
    if not text:
        raise EmptyDocumentError(
            "No extractable text found. Scanned PDFs need OCR, which is not supported."
        )
    return text, file_type
