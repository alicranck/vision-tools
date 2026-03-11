from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from vision_tools.stores.base import MemoryEntry

logger = logging.getLogger(__name__)

_DEFAULT_STORE_DIR = Path.home() / ".visionpilot" / "stores"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory_entries (
    id           TEXT PRIMARY KEY,
    store_id     TEXT NOT NULL,
    label        TEXT,
    media_uri    TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at   REAL NOT NULL,
    last_seen_at REAL NOT NULL
)
"""


class LocalMemoryBackend:
    """SQLite + in-memory numpy backend for MemoryStore.

    - SQLite stores metadata (label, media_uri, timestamps, arbitrary metadata JSON).
    - A numpy float32 matrix holds all embedding vectors in memory, persisted to disk as
      a .npy file after each write. Cosine similarity search is done in-process.
    - Media frames/crops are stored as files under {store_dir}/{store_id}/media/.

    Suitable for hundreds to low thousands of entries. For larger scale, swap in a
    Qdrant or PostgreSQL+pgvector backend via the MemoryBackend protocol.
    """

    def __init__(self, store_id: str, store_dir: Path | str | None = None) -> None:
        self._store_id = store_id
        root = Path(store_dir or _DEFAULT_STORE_DIR) / store_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "media").mkdir(exist_ok=True)

        self._db_path = root / "index.db"
        self._np_path = root / "embeddings.npy"
        self._ids_path = root / "embedding_ids.json"

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

        # Load embedding matrix into memory
        self._embeddings: np.ndarray | None = None  # shape (N, dim)
        self._embedding_ids: list[str] = []
        self._load_index()

    # ------------------------------------------------------------------
    # Public API (implements MemoryBackend protocol)
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> str:
        entry_id = entry.id or str(uuid.uuid4())
        now = time.time()

        self._conn.execute(
            """INSERT OR REPLACE INTO memory_entries
               (id, store_id, label, media_uri, metadata_json, created_at, last_seen_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id,
                self._store_id,
                entry.label,
                entry.media_uri,
                json.dumps(entry.metadata),
                entry.created_at or now,
                entry.last_seen_at or now,
            ),
        )
        self._conn.commit()

        vec = self._normalize(entry.embedding)
        if self._embeddings is None:
            self._embeddings = vec.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])
        self._embedding_ids.append(entry_id)
        self._save_index()

        logger.debug("memory_store_add", store_id=self._store_id, entry_id=entry_id)
        return entry_id

    def nearest_neighbors(self, vector: list[float], k: int) -> list[MemoryEntry]:
        if self._embeddings is None or not self._embedding_ids:
            return []

        vec = self._normalize(vector)
        similarities = self._embeddings @ vec  # cosine similarity (vecs are normalized)

        top_k = min(k, len(self._embedding_ids))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: list[MemoryEntry] = []
        for idx in top_indices:
            entry_id = self._embedding_ids[int(idx)]
            entry = self._fetch_entry(entry_id)
            if entry:
                entry.metadata["_similarity"] = float(similarities[int(idx)])
                results.append(entry)
        return results

    def query_and_add(self, entry: MemoryEntry, threshold: float) -> list[MemoryEntry]:
        if not entry.embedding:
            return []

        matches = self.nearest_neighbors(entry.embedding, k=1)
        if matches and matches[0].metadata.get("_similarity", 0.0) >= threshold:
            # Known entity — update last_seen_at
            self._conn.execute(
                "UPDATE memory_entries SET last_seen_at = ? WHERE id = ?",
                (time.time(), matches[0].id),
            )
            self._conn.commit()
            return matches
        else:
            # New entity — add to store
            self.add(entry)
            return []

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        if self._np_path.exists() and self._ids_path.exists():
            self._embeddings = np.load(str(self._np_path))
            with open(self._ids_path) as f:
                self._embedding_ids = json.load(f)
            logger.debug(
                "memory_index_loaded",
                store_id=self._store_id,
                entries=len(self._embedding_ids),
            )
        else:
            self._embeddings = None
            self._embedding_ids = []

    def _save_index(self) -> None:
        if self._embeddings is not None:
            np.save(str(self._np_path), self._embeddings)
        with open(self._ids_path, "w") as f:
            json.dump(self._embedding_ids, f)

    @staticmethod
    def _normalize(vector: list[float]) -> np.ndarray:
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def _fetch_entry(self, entry_id: str) -> MemoryEntry | None:
        row = self._conn.execute(
            "SELECT id, store_id, label, media_uri, metadata_json, created_at, last_seen_at "
            "FROM memory_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if not row:
            return None
        return MemoryEntry(
            id=row[0],
            store_id=row[1],
            label=row[2],
            media_uri=row[3],
            metadata=json.loads(row[4] or "{}"),
            created_at=row[5],
            last_seen_at=row[6],
            embedding=[],  # not stored in SQLite; only in numpy index
        )

    def entry_count(self) -> int:
        return len(self._embedding_ids)
