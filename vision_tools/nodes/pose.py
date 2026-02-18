"""
PoseEstimator — Task node for human pose estimation.

Usage::

    {"node_type": "pose_estimator",
     "config": {"model": "yolo_pose"}}
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.schemas import PoseResult
from vision_tools.nodes.model_node import ModelNode

logger = logging.getLogger(__name__)


class PoseEstimatorConfig(BaseModel):
    """Config schema for PoseEstimator."""
    model: str = Field("yolo_pose", description="Model backend: 'yolo_pose'")
    runtime: str = Field("auto", description="Runtime: 'auto', 'pytorch', 'openvino'")
    imgsz: int = Field(640, description="Input image size")
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")


@NodeRegistry.register("pose_estimator", category="pose")
class PoseEstimator(ModelNode):
    """Estimate human body keypoints in images.

    The model backend (YOLO-Pose, etc.) is resolved from config.
    """

    OutputSchema = PoseResult
    InputSchema = None

    def __init__(self, node_id: str = "pose_estimator", config: dict | None = None, **kwargs) -> None:
        config = config or {}
        model_name = config.get("model", "yolo_pose")
        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(task="pose", model=model_name)
        super().__init__(node_id=node_id, config=config, **kwargs)

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return PoseEstimatorConfig
