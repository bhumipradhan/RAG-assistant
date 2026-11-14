import pytest

from app.schemas.ingestion import ChunkingStrategy
from app.services.chunking.fixed import FixedSizeChunker
from app.services.chunking.recursive import RecursiveChunker
from app.services.chunking.registry import get_chunker

TEXT = ("Alpha paragraph sentence one. Sentence two here.\n\n" * 20).strip()


def test_fixed_respects_size_and_overlap() -> None:
    chunks = FixedSizeChunker(50, 10).chunk("x" * 120)
    assert [len(c) for c in chunks] == [50, 50, 40]


def test_fixed_no_tiny_duplicate_tail() -> None:
    assert len(FixedSizeChunker(50, 10).chunk("x" * 50)) == 1


@pytest.mark.parametrize("cls", [FixedSizeChunker, RecursiveChunker])
def test_invalid_params(cls: type[FixedSizeChunker] | type[RecursiveChunker]) -> None:
    with pytest.raises(ValueError):
        cls(0, 0)
    with pytest.raises(ValueError):
        cls(10, 10)


def test_recursive_chunks_within_size_and_cover_text() -> None:
    chunks = RecursiveChunker(120, 20).chunk(TEXT)
    assert all(len(c) <= 120 for c in chunks)
    assert len(chunks) > 1
    assert "Alpha paragraph" in chunks[0]


def test_recursive_hard_splits_when_no_separator() -> None:
    chunks = RecursiveChunker(30, 0).chunk("y" * 100)
    assert all(len(c) <= 30 for c in chunks)
    assert "".join(chunks) == "y" * 100


def test_registry_returns_each_strategy() -> None:
    assert isinstance(get_chunker(ChunkingStrategy.FIXED, 100, 10), FixedSizeChunker)
    assert isinstance(get_chunker(ChunkingStrategy.RECURSIVE, 100, 10), RecursiveChunker)
