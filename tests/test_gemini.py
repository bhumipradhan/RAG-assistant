from types import SimpleNamespace
from typing import Any, cast

import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import EmbeddingProvider, LLMProvider, Settings
from app.core.exceptions import ExternalServiceError
from app.schemas.chat import ChatMessage, Role
from app.services import retry
from app.services.embeddings import GeminiEmbedder, build_embedder
from app.services.llm import GeminiLLM, build_llm


class StubModels:
    """Stands in for `client.models`. `script` items are returned or raised in order."""

    def __init__(self, script: list[object]) -> None:
        self.script = script
        self.calls: list[dict[str, Any]] = []

    def _next(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def generate_content(self, **kwargs: Any) -> Any:
        return self._next(**kwargs)

    def embed_content(self, **kwargs: Any) -> Any:
        return self._next(**kwargs)


class StubClient:
    def __init__(self, script: list[object]) -> None:
        self.models = StubModels(script)


def client_of(script: list[object]) -> tuple[genai.Client, StubModels]:
    stub = StubClient(script)
    return cast(genai.Client, stub), stub.models


def reply(text: str | None) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ]
        if text is not None
        else []
    )


def embeddings(*vectors: list[float]) -> types.EmbedContentResponse:
    return types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=v) for v in vectors]
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace the `time` module only inside retry.py (patching time.sleep itself would be global).
    monkeypatch.setattr(retry, "time", SimpleNamespace(sleep=lambda _s: None))


def test_llm_maps_roles_and_passes_system_prompt() -> None:
    client, models = client_of([reply("hello")])
    llm = GeminiLLM("k", "gemini-2.5-flash", client)
    out = llm.complete(
        "be brief",
        [
            ChatMessage(role=Role.USER, content="hi"),
            ChatMessage(role=Role.ASSISTANT, content="hey"),
            ChatMessage(role=Role.USER, content="again"),
        ],
        max_tokens=5,
    )
    assert out == "hello"
    call = models.calls[0]
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["config"].system_instruction == "be brief"
    assert call["config"].max_output_tokens == 5  # thinking is off, so no headroom needed
    assert call["config"].thinking_config.thinking_budget == 0  # 2.5 flash: thinking off


def test_llm_leaves_headroom_for_thinking_on_newer_models() -> None:
    client, models = client_of([reply("hi")])
    GeminiLLM("k", "gemini-3.8-flash", client).complete("s", [], max_tokens=5)
    config = models.calls[0]["config"]
    assert config.max_output_tokens > 1000 and config.thinking_config is None


def test_llm_blocked_reply_returns_empty_string() -> None:
    client, _ = client_of([reply(None)])
    assert GeminiLLM("k", "gemini-2.5-flash", client).complete("s", []) == ""


def test_llm_retries_on_429_then_succeeds() -> None:
    err = genai_errors.ClientError(429, {"error": {"message": "slow down"}})
    client, models = client_of([err, err, reply("ok")])
    assert GeminiLLM("k", "m", client).complete("s", []) == "ok"
    assert len(models.calls) == 3


def test_llm_gives_up_and_translates_error() -> None:
    err = genai_errors.ClientError(429, {"error": {"message": "quota"}})
    client, models = client_of([err] * 4)
    with pytest.raises(ExternalServiceError, match="429"):
        GeminiLLM("k", "m", client).complete("s", [])
    assert len(models.calls) == 4


def test_llm_does_not_retry_bad_requests() -> None:
    err = genai_errors.ClientError(400, {"error": {"message": "bad key"}})
    client, models = client_of([err])
    with pytest.raises(ExternalServiceError):
        GeminiLLM("k", "m", client).complete("s", [])
    assert len(models.calls) == 1


def test_embedder_batches_normalizes_and_sets_task_types() -> None:
    texts = [f"t{i}" for i in range(60)]  # 60 > batch size of 50 => two API calls
    client, models = client_of([embeddings(*[[3.0, 4.0]] * 50), embeddings(*[[3.0, 4.0]] * 10)])
    emb = GeminiEmbedder("k", "gemini-embedding-001", 2, client)
    vectors = emb.embed_documents(texts)
    assert len(vectors) == 60 and vectors[0] == pytest.approx([0.6, 0.8])  # unit length
    assert models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert models.calls[0]["config"].output_dimensionality == 2

    client2, models2 = client_of([embeddings([1.0, 0.0])])
    assert GeminiEmbedder("k", "m", 2, client2).embed_query("q") == [1.0, 0.0]
    assert models2.calls[0]["config"].task_type == "RETRIEVAL_QUERY"


def test_embedder_detects_short_result() -> None:
    client, _ = client_of([embeddings([1.0, 0.0])])
    with pytest.raises(ExternalServiceError):
        GeminiEmbedder("k", "m", 2, client).embed_documents(["a", "b"])


def test_factories_select_gemini_and_require_key() -> None:
    no_key = Settings(
        llm_provider=LLMProvider.GEMINI,
        embedding_provider=EmbeddingProvider.GEMINI,
        _env_file=None,  # type: ignore[call-arg]
    )
    with pytest.raises(ExternalServiceError, match="GEMINI_API_KEY"):
        build_llm(no_key)
    with pytest.raises(ExternalServiceError, match="GEMINI_API_KEY"):
        build_embedder(no_key)

    with_key = Settings(
        llm_provider=LLMProvider.GEMINI,
        embedding_provider=EmbeddingProvider.GEMINI,
        gemini_api_key="fake-key",  # type: ignore[arg-type]
        _env_file=None,  # type: ignore[call-arg]
    )
    assert isinstance(build_llm(with_key), GeminiLLM)
    assert isinstance(build_embedder(with_key), GeminiEmbedder)


class TimeoutException(Exception):  # noqa: N818  (mimics the HTTP client's class name)
    pass


class ReadTimeout(TimeoutException):
    pass


def test_llm_timeout_is_retried_then_becomes_clean_error() -> None:
    client, models = client_of([ReadTimeout("timed out")] * 4)
    with pytest.raises(ExternalServiceError, match="timed out"):
        GeminiLLM("k", "m", client).complete("s", [])
    assert len(models.calls) == 4


def test_embedder_timeout_becomes_clean_error() -> None:
    client, _ = client_of([ReadTimeout("timed out")] * 4)
    with pytest.raises(ExternalServiceError, match="timed out"):
        GeminiEmbedder("k", "m", 2, client).embed_query("q")


def test_real_sdk_timeout_is_recognised() -> None:
    """End to end with the real SDK against a server that never answers."""
    import http.server
    import socketserver
    import threading
    import time

    from app.services.llm import gemini_client, is_timeout_error

    class Slow(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            time.sleep(3)

        def log_message(self, *args: object) -> None:
            pass

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = Server(("127.0.0.1", 0), Slow)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = genai.Client(
            api_key="x",
            http_options=types.HttpOptions(
                base_url=f"http://127.0.0.1:{server.server_address[1]}", timeout=500
            ),
        )
        with pytest.raises(Exception) as caught:
            client.models.generate_content(model="m", contents="hi")
        assert is_timeout_error(caught.value)
    finally:
        server.shutdown()
    assert gemini_client("k", 5) is not None
