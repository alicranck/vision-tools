from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from vision_tools.core.graph_types import Image
from vision_tools.core.node import NodeContext
from vision_tools.nodes.logic.logic_node import LogicNode

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_DIR = Path.home() / ".visionpilot" / "datasets"


class DatasetWriterConfig(BaseModel):
    dataset_id: str
    output_dir: str = ""  # Default: ~/.visionpilot/datasets/{dataset_id}/frames/
    annotation_hint: dict[str, Any] = {}


class DatasetWriterNode(LogicNode):
    """Write frames to a local dataset directory for later annotation and training.

    Each frame is saved as a JPEG alongside a JSON sidecar containing:
    - frame provenance (frame_idx, timestamp, camera_id, run context)
    - annotation_hint: pre-filled metadata to guide the annotation review UI
    - annotation_status: always 'pending' (awaiting user review)

    This is a sink node — it has no output ports.

    The dataset directory layout:
        ~/.visionpilot/datasets/{dataset_id}/
            frames/
                {uuid}.jpg
                {uuid}.json   ← sidecar

    VisionPilot's annotation review UI reads this directory to surface
    pending entries for labeling. (Integration deferred; files written now.)
    """

    InputPorts = {"image": "Image"}
    OutputPorts = {}  # sink node

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = DatasetWriterConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._config = validated

        root = Path(validated.output_dir or _DEFAULT_DATASET_DIR / validated.dataset_id)
        self._frames_dir = root / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> dict[str, Any]:
        image: Image = inputs["image"]
        entry_id = str(uuid.uuid4())

        # Save image
        img_path = self._frames_dir / f"{entry_id}.jpg"
        arr = np.asarray(image.data)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # RGB → BGR for cv2
        cv2.imwrite(str(img_path), arr)

        # Write sidecar
        sidecar = {
            "id": entry_id,
            "dataset_id": self._config.dataset_id,
            "annotation_status": "pending",
            "frame_idx": context.frame_idx,
            "timestamp": context.timestamp,
            "camera_id": context.camera_id,
            "frame_width": image.width,
            "frame_height": image.height,
            "captured_at": time.time(),
            "annotation_hint": self._config.annotation_hint,
        }
        sidecar_path = self._frames_dir / f"{entry_id}.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2))

        logger.debug(
            "dataset_writer_entry",
            dataset_id=self._config.dataset_id,
            entry_id=entry_id,
            frame_idx=context.frame_idx,
        )
        return {}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return DatasetWriterConfig
