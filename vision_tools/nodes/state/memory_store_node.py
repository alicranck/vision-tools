from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from vision_tools.core.config import MemoryOperation
from vision_tools.core.graph_types import Embedding, Image, MemoryMatch, MemoryMatches
from vision_tools.core.node import NodeContext
from vision_tools.core.type_refs import PortTypeRef, SimpleTypeRef
from vision_tools.nodes.logic.logic_node import LogicNode
from vision_tools.stores.base import MemoryEntry
from vision_tools.stores.local import LocalMemoryBackend

logger = logging.getLogger(__name__)


class MemoryStoreConfig(BaseModel):
    store_id: str = Field(description="Unique identifier for this memory store. Used to persist and reload the embedding index across runs.")
    operation: MemoryOperation = Field(MemoryOperation.QUERY_AND_ADD, description="Operation to perform each frame: 'add' stores the embedding, 'nearest_neighbors' returns top-k matches, 'query_and_add' searches first and only stores if no close match is found.")
    similarity_threshold: float = Field(0.75, ge=0.0, le=1.0, description="Cosine similarity cutoff for query_and_add: embeddings above this threshold are treated as a known match.", json_schema_extra={"x-ui-widget": "slider"})
    k: int = Field(1, ge=1, description="Number of nearest neighbours to return for 'nearest_neighbors' and 'query_and_add' operations.")
    with_image: bool = Field(False, description="If enabled, an 'image' input port is added and the image is saved alongside the embedding for later review.")
    store_dir: str = Field("", description="Override the default storage directory (~/.visionpilot/stores). Leave empty to use the default.", json_schema_extra={"x-advanced": True})


class MemoryStoreNode(LogicNode):
    """Runtime-queryable embedding knowledge bank.

    Backed by a LocalMemoryBackend (SQLite metadata + in-memory numpy ANN index).
    The index persists to disk across runs.

    Operations:
    - add: store the embedding (+ optional image). No output.
    - nearest_neighbors: return top-k closest entries.
    - query_and_add: search first; if best match >= threshold, return it;
        otherwise add as a new entry and return empty MemoryMatches.

    Use query_and_add for re-identification patterns (e.g., UC7 face re-ID):
    the node decides per-frame whether an entity is known or new.
    """

    DynamicPorts = True

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = MemoryStoreConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._config = validated
        self._backend: LocalMemoryBackend | None = None  # initialized at load time

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        store_dir = self._config.store_dir or None
        self._backend = LocalMemoryBackend(self._config.store_id, store_dir=store_dir)
        logger.info(
            "memory_store_loaded",
            store_id=self._config.store_id,
            entries=self._backend.entry_count(),
        )

    def unload(self) -> None:
        if self._backend:
            self._backend.close()
            self._backend = None

    # ------------------------------------------------------------------
    # Port declarations (dynamic — depend on operation + with_image)
    # ------------------------------------------------------------------

    def get_input_ports(self) -> dict[str, PortTypeRef]:
        ports: dict[str, PortTypeRef] = {"embedding": SimpleTypeRef("Embedding")}
        if self._config.with_image:
            ports["image"] = SimpleTypeRef("Image")
        return ports

    def get_output_ports(self) -> dict[str, PortTypeRef]:
        if self._config.operation == MemoryOperation.ADD:
            return {}  # write-only
        return {"matches": SimpleTypeRef("MemoryMatches")}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> dict[str, Any]:
        if self._backend is None:
            raise RuntimeError(f"{self.node_id}: backend not loaded — call load() first")

        embedding: Embedding = inputs["embedding"]

        # Empty embedding (e.g. no face detected this frame) — nothing to store or query.
        if not embedding.vector:
            if self._config.operation == "add":
                return {}
            return {"matches": MemoryMatches(items=[])}

        image: Image | None = inputs.get("image")

        media_uri: str | None = None
        if image is not None:
            media_uri = self._save_media(image, context)

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            store_id=self._config.store_id,
            embedding=embedding.vector,
            media_uri=media_uri,
            metadata={
                "frame_idx": context.frame_idx,
                "timestamp": context.timestamp,
                "camera_id": context.camera_id,
                "model_id": embedding.model_id,
            },
            created_at=time.time(),
            last_seen_at=time.time(),
        )

        op = self._config.operation
        if op == MemoryOperation.ADD:
            self._backend.add(entry)
            return {}

        elif op == MemoryOperation.NEAREST_NEIGHBORS:
            raw = self._backend.nearest_neighbors(embedding.vector, self._config.k)
            return {"matches": self._to_graph_type(raw)}

        else:  # query_and_add
            raw = self._backend.query_and_add(entry, self._config.similarity_threshold)
            return {"matches": self._to_graph_type(raw)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_media(self, image: Image, context: NodeContext) -> str:
        from vision_tools.stores.local import _DEFAULT_STORE_DIR

        store_dir = Path(self._config.store_dir or _DEFAULT_STORE_DIR)
        media_dir = store_dir / self._config.store_id / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{context.frame_idx}_{int(time.time() * 1000)}.jpg"
        path = media_dir / filename

        arr = np.asarray(image.data)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # RGB → BGR
        cv2.imwrite(str(path), arr)
        return str(path)

    @staticmethod
    def _to_graph_type(entries: list) -> MemoryMatches:
        items = [
            MemoryMatch(
                entry_id=e.id,
                label=e.label,
                similarity=max(
                    0.0,
                    min(1.0, float(e.metadata.get("_similarity", 0.0))),
                ),
                media_uri=e.media_uri,
                metadata={k: v for k, v in e.metadata.items() if k != "_similarity"},
            )
            for e in entries
        ]
        return MemoryMatches(items=items)

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return MemoryStoreConfig
