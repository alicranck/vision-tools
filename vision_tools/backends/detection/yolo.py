"""
YoloBackend — YOLO model loading, inference, postprocessing.

Ported from ``OpenVocabularyDetector``. Supports PyTorch and OpenVINO runtimes.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext
from vision_tools.core.schemas import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


@BackendRegistry.register(task="detection", model="yolo")
class YoloBackend:
    """Backend for YOLO-family object detection models.

    Supports:
    - PyTorch (CUDA or CPU)
    - OpenVINO (Intel CPU optimization)
    - ONNX (cross-platform)
    """

    def __init__(self) -> None:
        self._tracker = None
        self._kalman_filters: dict = {}
        self._tracking_history: dict = defaultdict(list)

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        """Load YOLO model and set up tracker.

        Args:
            model_path: Path to YOLO model file.
            device: Target device — "auto", "cuda", "cpu", "openvino".
        """
        from ultralytics import YOLOE
        from trackers import SORTTracker

        model = YOLOE(model_path)

        # Select runtime
        if device in ("openvino", "ov"):
            model = self._compile_openvino(model)
        elif device == "auto":
            # Auto: CUDA if available, else OpenVINO
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("YoloBackend: using CUDA")
                else:
                    model = self._compile_openvino(model)
            except ImportError:
                model = self._compile_openvino(model)

        self._tracker = SORTTracker(
            lost_track_buffer=5,
            frame_rate=10,
            minimum_consecutive_frames=2,
            minimum_iou_threshold=0.2,
        )

        return model

    def infer(self, model: Any, inputs: Any) -> Any:
        """Run YOLO inference with tracking."""
        import supervision as sv
        from vision_tools.utils.tracking import BoxKalmanFilter

        conf = 0.25  # default, can be overridden through config
        imgsz = 640

        results = model.predict(inputs, conf=conf, imgsz=imgsz)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = self._tracker.update(detections)

        # Update Kalman filters
        for i, track_id in enumerate(detections.tracker_id):
            if track_id is None or track_id == -1:
                continue
            if track_id not in self._kalman_filters:
                self._kalman_filters[track_id] = BoxKalmanFilter(
                    detections.xyxy[i],
                    detections.class_id[i],
                    detections.confidence[i],
                )
            else:
                self._kalman_filters[track_id].update(detections.xyxy[i])

        # Remove finished tracks
        active_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()
        finished = set(self._kalman_filters.keys()) - active_ids
        for ft_id in finished:
            del self._kalman_filters[ft_id]

        return {"tracks": self._kalman_filters, "class_names": results[0].names}

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        """Convert YOLO + tracker output to DetectionResult dict."""
        class_names = raw_output["class_names"]
        boxes = [
            BoundingBox(
                xyxy=list(map(float, kf.xyxy)),
                class_id=int(kf.class_idx),
                confidence=float(kf.conf),
                tracker_id=int(tid),
                class_name=class_names.get(int(kf.class_idx)),
            )
            for tid, kf in raw_output["tracks"].items()
        ]
        return DetectionResult(boxes=boxes, class_names=class_names).model_dump()

    def get_available_runtimes(self) -> list[str]:
        return ["pytorch", "openvino", "onnx"]

    @staticmethod
    def _compile_openvino(model) -> Any:
        """Export model to OpenVINO format."""
        logger.info("YoloBackend: compiling to OpenVINO")
        from ultralytics import YOLOE
        exported = model.export(
            format="openvino", simplify=True,
            nms=True, imgsz=640, batch=1, dynamic=True,
        )
        return YOLOE(exported)
