from typing import Protocol


class Chunker(Protocol):
    """Anything with a `chunk` method is a chunker (structural typing)."""

    def chunk(self, text: str) -> list[str]: ...
