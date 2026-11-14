"""Custom RAG pipeline (no RetrievalQA chain): rewrite, embed, retrieve, prompt, generate."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.chat import ChatMessage, Role, SourceOut
from app.services.embeddings import Embedder
from app.services.llm import LLMClient
from app.services.prompts import ANSWER_SYSTEM, REWRITE_SYSTEM
from app.services.vector_store import SearchHit, VectorStore

_NO_CONTEXT_REPLY = "I couldn't find anything relevant in the uploaded documents."
_HISTORY_FOR_REWRITE = 6
_SNIPPET_CHARS = 200


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[SourceOut]


class RagService:
    def __init__(
        self,
        *,
        llm: LLMClient,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        min_score: float,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def answer(self, question: str, history: Sequence[ChatMessage]) -> RagAnswer:
        standalone = self._standalone_question(question, history)
        hits = [
            h
            for h in self._store.search(self._embedder.embed_query(standalone), self._top_k)
            if h.score >= self._min_score
        ]
        if not hits:
            return RagAnswer(_NO_CONTEXT_REPLY, [])

        prompt = f"Context:\n{self._format_context(hits)}\n\nQuestion: {question}"
        messages = [*history, ChatMessage(role=Role.USER, content=prompt)]
        reply = self._llm.complete(ANSWER_SYSTEM, messages, temperature=0.2, max_tokens=800)
        return RagAnswer(reply.strip(), [self._to_source(h) for h in hits])

    def _standalone_question(self, question: str, history: Sequence[ChatMessage]) -> str:
        """'What about its pricing?' embeds terribly alone, so resolve it against history first."""
        if not history:
            return question
        recent = list(history[-_HISTORY_FOR_REWRITE:])
        messages = [*recent, ChatMessage(role=Role.USER, content=question)]
        rewritten = self._llm.complete(REWRITE_SYSTEM, messages, max_tokens=200).strip()
        return rewritten or question

    @staticmethod
    def _format_context(hits: Sequence[SearchHit]) -> str:
        return "\n\n".join(f"[{i}] ({h.filename}) {h.text}" for i, h in enumerate(hits, start=1))

    @staticmethod
    def _to_source(hit: SearchHit) -> SourceOut:
        return SourceOut(
            document_id=hit.document_id,
            filename=hit.filename,
            chunk_index=hit.chunk_index,
            score=round(hit.score, 4),
            snippet=hit.text[:_SNIPPET_CHARS],
        )
