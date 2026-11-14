"""Find which Gemini chat models respond right now with YOUR key.

Free-tier models are often overloaded (HTTP 503) or retired for new users (404), and the
names change. Run from the project root:   python -m scripts.check_models
Then put the fastest "OK" model in .env as LLM_MODEL.
"""

import time

from google import genai
from google.genai import errors, types

from app.core.config import get_settings

_SKIP = ("image", "tts", "audio", "live", "embedding", "robotics", "native", "veo", "lyria")
_MAX_MODELS = 12
_TIMEOUT_MS = 20_000


def candidate_models(client: genai.Client) -> list[str]:
    names: list[str] = []
    for model in client.models.list():
        name = (model.name or "").removeprefix("models/")
        if "generateContent" not in (model.supported_actions or []):
            continue
        if "flash" in name and not any(word in name for word in _SKIP):
            names.append(name)
    return sorted(set(names))[:_MAX_MODELS]


def main() -> None:
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise SystemExit("GEMINI_API_KEY is missing in .env")
    client = genai.Client(
        api_key=settings.gemini_api_key.get_secret_value(),
        http_options=types.HttpOptions(timeout=_TIMEOUT_MS),
    )
    models = candidate_models(client)
    print(f"Currently configured LLM_MODEL: {settings.llm_model}")
    print(f"Testing {len(models)} flash models (20s timeout each)...\n")
    working: list[tuple[float, str]] = []
    for name in models:
        start = time.time()
        try:
            client.models.generate_content(model=name, contents="Reply with the word: hi")
            elapsed = time.time() - start
            working.append((elapsed, name))
            print(f"  OK    {elapsed:5.1f}s  {name}")
        except errors.APIError as exc:
            print(f"  FAIL  {time.time() - start:5.1f}s  {name}  ({exc.code})")
        except Exception as exc:
            print(f"  FAIL  {time.time() - start:5.1f}s  {name}  ({type(exc).__name__})")
    if working:
        print(f"\nFastest working model: LLM_MODEL={min(working)[1]}")
    else:
        print("\nNone responded. Wait a few minutes and run again (503s are usually temporary).")


if __name__ == "__main__":
    main()
