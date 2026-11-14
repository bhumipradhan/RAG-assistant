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


def is_timeout_error(exc: BaseException) -> bool:
    """True for HTTP client timeouts. Checked by class name because the Google SDK may use
    either `httpx` or `httpx2` underneath, and both name it TimeoutException."""
    return any(cls.__name__ == "TimeoutException" for cls in type(exc).__mro__)


def is_retryable_gemini_error(exc: Exception) -> bool:
    """Rate limits (429), transient server errors (500/503) and timeouts are worth retrying."""
    if isinstance(exc, genai_errors.APIError):
        return exc.code in (429, 500, 503)
    return is_timeout_error(exc)


def gemini_client(api_key: str, timeout_seconds: int) -> genai.Client:
    """A client that gives up on a stalled request instead of waiting forever."""
    return genai.Client(
        api_key=api_key, http_options=genai_types.HttpOptions(timeout=timeout_seconds * 1000)
    )


# Models that cannot switch "thinking" off spend output tokens on it, so leave room for it.
_THINKING_HEADROOM = 2048


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: genai.Client | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self._client = client or gemini_client(api_key, timeout_seconds)
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
        can_skip_thinking = "2.5-flash" in self._model
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            # "Thinking" tokens count towards the output limit, so a small limit (such as the
            # 5 tokens used for intent labels) could produce an empty or cut-off reply.
            max_output_tokens=max_tokens + (0 if can_skip_thinking else _THINKING_HEADROOM),
            # 2.5 Flash can skip thinking entirely: faster and fewer rate-limit hits.
            thinking_config=(
                genai_types.ThinkingConfig(thinking_budget=0) if can_skip_thinking else None
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
        except Exception as exc:
            if is_timeout_error(exc):
                raise ExternalServiceError(
                    "The LLM provider timed out. Try again shortly."
                ) from exc
            raise
        return response.text or ""  # text is None when the reply was blocked or empty


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider is LLMProvider.GEMINI:
        if settings.gemini_api_key is None:
            raise ExternalServiceError("GEMINI_API_KEY is required.")
        return GeminiLLM(
            settings.gemini_api_key.get_secret_value(),
            settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if settings.llm_provider is LLMProvider.ANTHROPIC:
        if settings.anthropic_api_key is None:
            raise ExternalServiceError("ANTHROPIC_API_KEY is required.")
        return AnthropicLLM(settings.anthropic_api_key.get_secret_value(), settings.llm_model)
    if settings.openai_api_key is None:
        raise ExternalServiceError("OPENAI_API_KEY is required.")
    return OpenAILLM(settings.openai_api_key.get_secret_value(), settings.llm_model)
