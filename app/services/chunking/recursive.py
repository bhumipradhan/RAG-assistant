_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


class RecursiveChunker:
    """Split on the biggest natural boundary that fits (paragraph > line > sentence > word),
    then merge neighbouring pieces back up to `chunk_size`, carrying `overlap` characters of
    whole pieces into the next chunk. Keeps sentences/paragraphs intact where possible.
    """

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self._size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str) -> list[str]:
        return self._merge(self._split(text, _SEPARATORS))

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self._size:
            return [text]
        if not separators:  # no boundary left: hard cut
            return [text[i : i + self._size] for i in range(0, len(text), self._size)]
        sep, rest = separators[0], separators[1:]
        parts = text.split(sep)
        pieces: list[str] = []
        for i, part in enumerate(parts):
            piece = part + sep if i < len(parts) - 1 else part
            pieces.extend(self._split(piece, rest))
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for piece in pieces:
            if current and length + len(piece) > self._size:
                chunks.append("".join(current).strip())
                # keep only a small tail of the previous chunk as overlap
                while current and (length > self._overlap or length + len(piece) > self._size):
                    length -= len(current.pop(0))
            current.append(piece)
            length += len(piece)
        if current:
            chunks.append("".join(current).strip())
        return [c for c in chunks if c]
