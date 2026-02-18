"""
YoloPoseBackend — YOLO-Pose model loading, inference, postprocessing.

Ported from ``PoseEstimator``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext
from vision_tools.core.schemas import Keypoint, PoseKeypoints, PoseResult

logger = logging.getLogger(__name__)


@BackendRegistry.register(task="pose", model="yolo_pose")
class YoloPoseBackend:
    """Backend for YOLO-Pose human pose estimation."""

    def __init__(self) -> None:
        self._imgsz = 640
        self._conf_threshold = 0.5

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        """Load YOLO-Pose model."""
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model

    def infer(self, model: Any, inputs: Any) -> Any:
        """Run YOLO-Pose inference."""
        results = model.predict(
            inputs,
            conf=self._conf_threshold,
            imgsz=self._imgsz,
            verbose=False,
        )
        return results[0]

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        """Convert YOLO-Pose results to PoseResult dict."""
        keypoints = raw_output.keypoints

        poses = []
        if keypoints is not None:
            for i, kpt in enumerate(keypoints):
                xy = kpt.xy[0].cpu().numpy().tolist()
                conf = (
                    kpt.conf[0].cpu().numpy().tolist()
                    if kpt.conf is not None
                    else [1.0] * 17
                )
                kpts = [
                    Keypoint(x=pt[0], y=pt[1], confidence=c)
                    for pt, c in zip(xy, conf)
                ]
                poses.append(PoseKeypoints(person_id=i, keypoints=kpts))

        return PoseResult(poses=poses).model_dump()

    def get_available_runtimes(self) -> list[str]:
        return ["pytorch", "openvino"]
