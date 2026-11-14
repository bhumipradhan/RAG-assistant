"""LLM providers behind one small interface."""

from collections.abc import Sequence
from typing import Protocol

import anthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from app.core.config import LLMProvider, Settings
from app.core.exceptions import ExternalServiceError
from app.schemas.chat import ChatMessage, Role
from app.services.retry import retry_call


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


def is_retryable_gemini_error(exc: Exception) -> bool:
    """Rate limit (429) and transient server errors are worth retrying."""
    return isinstance(exc, genai_errors.APIError) and exc.code in (429, 500, 503)


class GeminiLLM:
    def __init__(self, api_key: str, model: str, client: genai.Client | None = None) -> None:
        self._client = client or genai.Client(api_key=api_key)
        self._model = model

    def complete(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        contents = [
            genai_types.Content(
                # Gemini calls the assistant role "model".
                role="user" if m.role is Role.USER else "model",
                parts=[genai_types.Part(text=m.content)],
            )
            for m in messages
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            # Floor: "thinking" tokens count towards the limit on some models, and a tiny
            # limit (e.g. the 5 tokens used for intent labels) could return an empty reply.
            max_output_tokens=max(max_tokens, 512),
            # 2.5 Flash models can skip "thinking": faster, cheaper, fewer rate-limit hits.
            thinking_config=(
                genai_types.ThinkingConfig(thinking_budget=0)
                if "2.5-flash" in self._model
                else None
            ),
        )
        try:
            response = retry_call(
                lambda: self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                ),
                should_retry=is_retryable_gemini_error,
            )
        except genai_errors.APIError as exc:
            raise ExternalServiceError(f"LLM provider error ({exc.code}): {exc.message}") from exc
        return response.text or ""  # text is None when the reply was blocked or empty


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider is LLMProvider.GEMINI:
        if settings.gemini_api_key is None:
            raise ExternalServiceError("GEMINI_API_KEY is required.")
        return GeminiLLM(settings.gemini_api_key.get_secret_value(), settings.llm_model)
    if settings.llm_provider is LLMProvider.ANTHROPIC:
        if settings.anthropic_api_key is None:
            raise ExternalServiceError("ANTHROPIC_API_KEY is required.")
        return AnthropicLLM(settings.anthropic_api_key.get_secret_value(), settings.llm_model)
    if settings.openai_api_key is None:
        raise ExternalServiceError("OPENAI_API_KEY is required.")
    return OpenAILLM(settings.openai_api_key.get_secret_value(), settings.llm_model)
