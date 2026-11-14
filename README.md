# RAG Assistant: Document Ingestion + Conversational RAG + Interview Booking

Two REST APIs built with FastAPI:

1. **Document Ingestion API**: upload `.pdf` / `.txt`, extract text, chunk (2 selectable strategies), embed, store vectors in **Qdrant**, store metadata in **SQL** (SQLite by default, Postgres-ready).
2. **Conversational RAG API**: a hand-written RAG pipeline (no `RetrievalQAChain`), **Redis** chat memory, multi-turn follow-ups, and LLM-assisted **interview booking** (name, email, date, time) persisted to SQL.

No FAISS, no Chroma, no UI.

---

## 1. How it works (the mental model)

```
INGESTION                                    CHAT
upload file                                  user message ──► load history (Redis)
   │                                                │
   ▼                                                ▼
extract text (pypdf / utf-8)              classify intent (LLM) ── QUESTION ──► RAG
   │                                                │                           │
   ▼                                              BOOKING                 rewrite follow-up
chunk (fixed | recursive)                           │                      to standalone query
   │                                                ▼                           │
   ▼                                     extract fields (LLM)              embed + search Qdrant
embed (OpenAI | fastembed)                validate IN CODE                      │
   │                                     state in Redis                  prompt + context + history
   ▼                                     ask missing → confirm → save          │
Qdrant (vectors + chunk text)                       │                          ▼
SQL (document metadata)                             ▼                       answer + sources
                                              bookings table
                                   both paths ──► save turn to Redis (capped, TTL)
```

## 2. Project layout

```
app/
  main.py                    app factory, error handler, routers
  core/
    config.py                typed settings (pydantic-settings, env-driven)
    exceptions.py            domain errors -> HTTP status
    dependencies.py          dependency wiring (swap anything in tests)
  api/routes/                THIN routes: ingestion.py, chat.py
  schemas/                   Pydantic models: ingestion, chat, booking
  services/
    extraction.py            pdf/txt -> clean text
    chunking/                Chunker Protocol + fixed + recursive + registry
    embeddings.py            Embedder Protocol + OpenAI + FastEmbed (local)
    vector_store.py          VectorStore Protocol + Qdrant implementation
    llm.py                   LLMClient Protocol + OpenAI + Anthropic
    ingestion.py             ingestion pipeline (idempotent, compensating)
    memory.py                Redis chat history + booking state
    prompts.py               all prompt text in one place
    intent.py                BOOKING vs QUESTION classifier
    rag.py                   custom RAG: rewrite -> retrieve -> generate
    booking.py               slot filling, validation, confirmation, persistence
    chat.py                  chat orchestration
  db/                        SQLAlchemy 2.0 models, session, repositories
tests/                       25 tests, fully offline (fake LLM/embedder, in-memory Qdrant & Redis)
```

## 3. Run it

**Prerequisites:** Python 3.12+, Docker, an API key (OpenAI and/or Anthropic).

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 2. Configure
cp .env.example .env        # then fill in your key(s); see "Providers" below

# 3. Start Qdrant + Redis
docker compose up -d qdrant redis

# 4. Start the API
uvicorn app.main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

Or everything in Docker: `docker compose up --build`.

### Providers

| You have | Set in `.env` |
|---|---|
| OpenAI key only | `OPENAI_API_KEY=...` (defaults work) |
| Anthropic key only | `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-sonnet-5-5`, `ANTHROPIC_API_KEY=...`, plus local embeddings: `pip install fastembed`, `EMBEDDING_PROVIDER=fastembed`, `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, `EMBEDDING_DIMENSION=384` (Anthropic has no embeddings API) |

> `EMBEDDING_DIMENSION` must match the embedding model. If you change models, use a new `QDRANT_COLLECTION` name.

## 4. Try it

```bash
# Upload (strategy: fixed | recursive)
curl -F "file=@notes.txt" -F "strategy=recursive" http://127.0.0.1:8000/api/v1/documents

# Ask a question
curl -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "What does the document say about pricing?"}'

# Follow-up (same session_id; memory resolves "it")
curl -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "and how long does it take?"}'

# Book an interview (multi-turn, same session_id)
curl -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "I want to book an interview"}'
```

## 5. Tests and quality gates

```bash
pytest -q          # 25 tests, no network, no API keys needed
mypy app           # strict typing
ruff check .       # lint
```

## 6. Design decisions (know these for the interview)

| Decision | Why | Trade-off |
|---|---|---|
| Sync `def` endpoints | FastAPI runs them in a threadpool, so blocking PDF/embedding/DB calls don't stall the event loop | Not as scalable as a fully async stack; fine for this workload |
| `Protocol` interfaces for LLM / Embedder / VectorStore / Chunker | Swap providers with no changes to business logic; trivial to fake in tests | Slightly more files |
| Chunker registry (Strategy pattern) | A third strategy = 1 class + 1 line | None really |
| Idempotency via SHA-256 of file bytes + strategy | Re-uploading doesn't duplicate vectors | Same file with a different strategy is intentionally a new document |
| Deterministic vector IDs (`uuid5(doc_id:chunk_index)`) | Retries overwrite, never duplicate | n/a |
| Compensating delete if SQL write fails after vector upsert | Avoids orphan vectors without a distributed transaction | Not atomic; a crash between steps could still leave orphans |
| Follow-up query rewriting | "What about its price?" is useless to embed on its own | One extra LLM call on follow-up turns |
| Booking state stored separately from chat history | History is capped/truncated; booking state must not be | Two Redis keys per session |
| LLM extracts, **code validates** (email, date, time, past checks, slot uniqueness) | The LLM can hallucinate; never trust it for correctness | More code than "let the LLM handle it" |
| Explicit confirmation step before saving a booking | Prevents wrong bookings from misparsed input | One extra turn |
| Unique constraint on (date, time) | Race-safe double-booking protection enforced by the DB | One interviewer assumed |
| Redis errors -> HTTP 503 | Clear failure mode instead of a stack trace | No fallback memory (stateless degraded mode would be a future improvement) |

## 7. Known limitations (honest list)

- Scanned/image PDFs are rejected (no OCR).
- Chunking is character-based, not token-based.
- Intent detection and extraction rely on prompting. Native tool/function calling would be a more robust upgrade.
- Times are interpreted in `TIMEZONE` (default UTC), not per-user timezones.
- No auth, rate limiting, or Alembic migrations (`create_all` is used at startup).
- No emails are actually sent on booking.
- Chat history truncates at the last N messages (no summarization).
- No endpoint to delete a document (the `delete_document` store method exists for it).
