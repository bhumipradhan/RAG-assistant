from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    body: ChatRequest, service: Annotated[ChatService, Depends(get_chat_service)]
) -> ChatResponse:
    return service.chat(body.session_id, body.message)
