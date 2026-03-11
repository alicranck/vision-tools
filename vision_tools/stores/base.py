from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MemoryEntry:
    """A single entry in a MemoryStore."""

    store_id: str
    embedding: list[float]
    id: str = ""
    label: str | None = None
    media_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_seen_at: float = 0.0


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for MemoryStore backends. Implement this to add new storage targets."""

    def add(self, entry: MemoryEntry) -> str:
        """Persist an entry. Returns the assigned entry ID."""
        ...

    def nearest_neighbors(self, vector: list[float], k: int) -> list[MemoryEntry]:
        """Return up to k entries closest to vector by cosine similarity."""
        ...

    def query_and_add(self, entry: MemoryEntry, threshold: float) -> list[MemoryEntry]:
        """Search for matches above threshold.

        If a match is found: update its last_seen_at and return it.
        If no match: add as a new entry and return an empty list.
        """
        ...

    def close(self) -> None:
        """Release resources (close DB connections, flush index, etc.)."""
        ...
