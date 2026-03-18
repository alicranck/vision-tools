from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from vision_tools.core.graph_types import History, Image
from vision_tools.core.node import NodeContext
from vision_tools.core.type_refs import GenericTypeRef, PortTypeRef, SimpleTypeRef
from vision_tools.nodes.logic.logic_node import LogicNode

logger = logging.getLogger(__name__)


class FileWriterConfig(BaseModel):
    output_dir: str = Field(description="Directory where output files are written. Created automatically if it does not exist.")
    media_type: str = Field("frame", pattern="^(frame|video_clip)$", description="Write mode: 'frame' saves individual JPEG/PNG images; 'video_clip' encodes a History[Image] buffer as an MP4.")
    filename_template: str = Field("{timestamp:.3f}_{frame_idx}", description="Template for the output filename (without extension). Supports {timestamp}, {frame_idx}, {camera_id}, and {ts} placeholders.", json_schema_extra={"x-advanced": True})
    fps: int = Field(30, gt=0, description="Frames per second for video_clip encoding.", json_schema_extra={"x-advanced": True})
    format: str = Field("", description="File extension override (e.g. 'jpg', 'png', 'mp4'). Leave empty to use the default for the chosen media_type.", json_schema_extra={"x-advanced": True})


class FileWriterNode(LogicNode):
    """Write frames or video clips to disk on demand.

    - media_type='frame': accepts a single Image and writes a JPEG/PNG.
    - media_type='video_clip': accepts History[Image] and encodes an MP4.

    This is a sink node — it has no output ports.
    The node is typically placed at the end of a triggered branch
    (e.g., after a fall-detection DynamicLogicNode).
    """

    DynamicPorts = True

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = FileWriterConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._config = validated
        self._out_dir = Path(validated.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def get_input_ports(self) -> dict[str, PortTypeRef]:
        if self._config.media_type == "frame":
            return {"image": SimpleTypeRef("Image")}
        else:
            return {"frames": GenericTypeRef("History", SimpleTypeRef("Image"))}

    def get_output_ports(self) -> dict[str, PortTypeRef]:
        return {}  # sink node — no outputs

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> dict[str, Any]:
        if self._config.media_type == "frame":
            self._write_frame(inputs["image"], context)
        else:
            self._write_clip(inputs["frames"], context)
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filename(self, context: NodeContext, extension: str) -> Path:
        name = self._config.filename_template.format(
            timestamp=context.timestamp,
            frame_idx=context.frame_idx,
            camera_id=context.camera_id,
            ts=int(time.time()),
        )
        return self._out_dir / f"{name}.{extension}"

    def _write_frame(self, image: Image, context: NodeContext) -> None:
        ext = self._config.format or "jpg"
        path = self._filename(context, ext)
        data = self._to_bgr(image.data)
        ok = cv2.imwrite(str(path), data)
        if not ok:
            logger.warning("file_writer_frame_failed", path=str(path), node_id=self.node_id)
        else:
            logger.debug("file_writer_frame_written", path=str(path))

    def _write_clip(self, history: History, context: NodeContext) -> None:
        frames = history.items
        if not frames:
            logger.debug("file_writer_clip_empty", node_id=self.node_id)
            return

        ext = self._config.format or "mp4"
        path = self._filename(context, ext)

        first: Image = frames[0]
        h, w = first.height, first.width
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self._config.fps, (w, h))

        for frame in frames:
            bgr = self._to_bgr(frame.data)
            # Resize if frames differ in size (shouldn't normally happen)
            if bgr.shape[0] != h or bgr.shape[1] != w:
                bgr = cv2.resize(bgr, (w, h))
            writer.write(bgr)

        writer.release()
        logger.debug("file_writer_clip_written", path=str(path), frames=len(frames))

    @staticmethod
    def _to_bgr(data: Any) -> np.ndarray:
        """Convert image data to uint8 BGR numpy array for cv2."""
        arr = np.asarray(data)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # RGB → BGR
        return arr

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return FileWriterConfig
