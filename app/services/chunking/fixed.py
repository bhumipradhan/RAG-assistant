class FixedSizeChunker:
    """Sliding window of `chunk_size` characters, each overlapping the previous by `overlap`.

    Simple and predictable, but can cut sentences in half.
    """

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self._size = chunk_size
        self._step = chunk_size - overlap

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._size
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start += self._step
        return chunks
