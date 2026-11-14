"""LLM providers behind one small interface."""

from collections.abc import Sequence
from typing import Protocol

import anthropic
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from app.core.config import LLMProvider, Settings
from app.core.exceptions import ExternalServiceError
from app.schemas.chat import ChatMessage, Role


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str: ...


class OpenAILLM:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        payload: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role is Role.USER:
                payload.append({"role": "user", "content": m.content})
            else:
                payload.append({"role": "assistant", "content": m.content})
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except OpenAIError as exc:
            raise ExternalServiceError(f"LLM provider error: {exc}") from exc
        return resp.choices[0].message.content or ""


class AnthropicLLM:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        payload: list[anthropic.types.MessageParam] = []
        for m in messages:
            if m.role is Role.USER:
                payload.append({"role": "user", "content": m.content})
            else:
                payload.append({"role": "assistant", "content": m.content})
        try:
            resp = self._client.messages.create(
                model=self._model,
                system=system,
                messages=payload,
                max_tokens=max_tokens,  # `temperature` is intentionally not sent (not accepted)
            )
        except anthropic.AnthropicError as exc:
            raise ExternalServiceError(f"LLM provider error: {exc}") from exc
        return "".join(b.text for b in resp.content if b.type == "text")


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider is LLMProvider.ANTHROPIC:
        if settings.anthropic_api_key is None:
            raise ExternalServiceError("ANTHROPIC_API_KEY is required.")
        return AnthropicLLM(settings.anthropic_api_key.get_secret_value(), settings.llm_model)
    if settings.openai_api_key is None:
        raise ExternalServiceError("OPENAI_API_KEY is required.")
    return OpenAILLM(settings.openai_api_key.get_secret_value(), settings.llm_model)
