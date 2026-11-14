from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile

from app.core.dependencies import DbDep, get_ingestion_service
from app.db.repositories import DocumentRepository
from app.schemas.ingestion import ChunkingStrategy, DocumentOut
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/documents", tags=["ingestion"])


# Plain `def` (not `async def`): FastAPI runs it in a thread pool, so blocking work
# (PDF parsing, embedding HTTP calls, DB writes) does not freeze the event loop.
@router.post("", response_model=DocumentOut, status_code=201)
def upload_document(
    file: Annotated[UploadFile, File(description=".pdf or .txt file")],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    response: Response,
    strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.RECURSIVE,
) -> DocumentOut:
    result = service.ingest(file.filename or "unnamed", file.file, strategy)
    if result.duplicate:
        response.status_code = 200
    return DocumentOut.model_validate(result.document).model_copy(
        update={"duplicate": result.duplicate}
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(db: DbDep) -> list[DocumentOut]:
    return [DocumentOut.model_validate(d) for d in DocumentRepository(db).list_all()]
