from collections.abc import Callable

from app.schemas.ingestion import ChunkingStrategy
from app.services.chunking.base import Chunker
from app.services.chunking.fixed import FixedSizeChunker
from app.services.chunking.recursive import RecursiveChunker

ChunkerFactory = Callable[[int, int], Chunker]

# To add a strategy: write a class, add an enum member, add ONE line here.
_REGISTRY: dict[ChunkingStrategy, ChunkerFactory] = {
    ChunkingStrategy.FIXED: FixedSizeChunker,
    ChunkingStrategy.RECURSIVE: RecursiveChunker,
}


def get_chunker(strategy: ChunkingStrategy, chunk_size: int, overlap: int) -> Chunker:
    return _REGISTRY[strategy](chunk_size, overlap)
