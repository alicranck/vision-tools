"""
ObjectDetector — Task node for object detection.

This is the "what": detect objects in an image. The "how" (YOLO, DETR, etc.)
is handled by the backend resolved from config.

Usage::

    {"node_type": "object_detector",
     "config": {"task": "detection", "model_family": "yolo",
                "size": "small", "device": "cpu",
                "vocabulary": ["person", "car"]}}
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.config import DeviceTarget, InferenceTask, ModelIntent, ModelSize
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.schemas import DetectionResult
from vision_tools.nodes.model_node import ModelNode
from vision_tools.runtime.model_catalog import ModelCatalog

logger = logging.getLogger(__name__)


class ObjectDetectorConfig(BaseModel):
    """Config schema for ObjectDetector — used by LLM agents for discovery."""
    task: InferenceTask = Field(
        default=InferenceTask.DETECTION,
        description="Task: 'detection' or 'open_vocab_detection'",
    )
    model_family: str = Field("yolo", description="Model family: 'yolo', 'rtdetr'")
    size: ModelSize = Field(ModelSize.SMALL, description="Model size tier")
    device: DeviceTarget = Field(DeviceTarget.AUTO, description="Execution target")
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
        config = ObjectDetectorConfig.model_validate(config or {})
        intent = ModelIntent(
            task=config.task,
            model_family=config.model_family,
            size=config.size,
            device=config.device,
        )
        resolved = ModelCatalog.resolve(intent)

        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(
                task=resolved.backend_task,
                model=resolved.backend_model,
            )

        resolved_config = config.model_dump(mode="json")
        resolved_config.update(
            {
                "runtime": resolved.runtime,
                "checkpoint_id": resolved.checkpoint_id,
                "backend_task": resolved.backend_task,
                "backend_model": resolved.backend_model,
            }
        )

        if hasattr(kwargs["backend"], "configure"):
            kwargs["backend"].configure(resolved_config)
        super().__init__(node_id=node_id, config=resolved_config, **kwargs)

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return ObjectDetectorConfig

    @classmethod
    def get_config_options(cls) -> dict[str, Any]:
        return {"model_catalog": ModelCatalog.list_options()}
