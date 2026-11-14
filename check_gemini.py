import time

from google import genai
from google.genai import types

from app.core.config import get_settings

s = get_settings()
assert s.gemini_api_key, "GEMINI_API_KEY is missing in .env"
client = genai.Client(
    api_key=s.gemini_api_key.get_secret_value(),
    http_options=types.HttpOptions(timeout=30_000),  # milliseconds
)


def timed(label, fn):
    start = time.time()
    try:
        print(f"{label}: OK in {time.time() - start:.1f}s -> {fn()}")
    except Exception as exc:
        print(f"{label}: FAILED after {time.time() - start:.1f}s -> {type(exc).__name__}: {str(exc)[:300]}")


timed(
    "embedding",
    lambda: len(
        client.models.embed_content(
            model=s.embedding_model,
            contents="hello",
            config=types.EmbedContentConfig(output_dimensionality=s.embedding_dimension),
        ).embeddings[0].values
    ),
)
timed(
    f"{s.llm_model} default",
    lambda: client.models.generate_content(model=s.llm_model, contents="Say hi").text,
)
timed(
    f"{s.llm_model} thinking=MINIMAL",
    lambda: client.models.generate_content(
        model=s.llm_model,
        contents="Say hi",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
        ),
    ).text,
)