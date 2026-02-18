"""
ObjectDetector — Task node for object detection.

This is the "what": detect objects in an image. The "how" (YOLO, DETR, etc.)
is handled by the backend resolved from config.

Usage::

    {"node_type": "object_detector",
     "config": {"model": "yolo", "vocabulary": ["person", "car"]}}
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.schemas import DetectionResult
from vision_tools.nodes.model_node import ModelNode

logger = logging.getLogger(__name__)


class ObjectDetectorConfig(BaseModel):
    """Config schema for ObjectDetector — used by LLM agents for discovery."""
    model: str = Field("yolo", description="Model backend: 'yolo'")
    runtime: str = Field("auto", description="Runtime: 'auto', 'pytorch', 'openvino', 'onnx'")
    vocabulary: list[str] = Field(default_factory=list, description="Classes to detect")
    imgsz: int = Field(640, description="Input image size")
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    prompt_free: bool = Field(False, description="Use prompt-free model")


@NodeRegistry.register("object_detector", category="detection")
class ObjectDetector(ModelNode):
    """Detect objects in images.

    Supports open-vocabulary detection with custom class lists.
    The model backend (YOLO, DETR, etc.) is resolved from config.
    """

    OutputSchema = DetectionResult
    InputSchema = None  # Raw frame input

    def __init__(self, node_id: str = "detector", config: dict | None = None, **kwargs) -> None:
        config = config or {}
        # Resolve backend from config
        model_name = config.get("model", "yolo")
        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(task="detection", model=model_name)
        if hasattr(kwargs["backend"], "configure"):
            kwargs["backend"].configure(config)
        super().__init__(node_id=node_id, config=config, **kwargs)

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return ObjectDetectorConfig
