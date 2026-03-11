"""
YoloBackend — YOLO model loading, inference, postprocessing.

Supports open-vocabulary detection inference for ``open_vocab_detector``.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.graph_types import BoundingBox, Detections
from vision_tools.core.node import NodeContext

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
        self._imgsz = 640
        self._conf_threshold = 0.25
        self._vocabulary: list[str] = []
        self._prompt_free = False

    def configure(self, config: dict[str, Any]) -> None:
        """Apply task-level config from ObjectDetector node."""
        self._imgsz = int(config.get("imgsz", self._imgsz))
        self._conf_threshold = float(config.get("conf_threshold", self._conf_threshold))
        self._vocabulary = list(config.get("vocabulary", self._vocabulary) or [])
        self._prompt_free = bool(config.get("prompt_free", self._prompt_free))

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        """Load YOLO model and set up tracker.

        Args:
            model_path: Path to YOLO model file.
            device: Target device — "auto", "cuda", "cpu", "openvino".
        """
        from ultralytics import YOLOE, settings as yolo_settings
        from trackers import SORTTracker

        model = YOLOE(self._resolve_checkpoint_path(model_path, yolo_settings))

        if not self._prompt_free and self._vocabulary:
            if hasattr(model, "get_text_pe") and hasattr(model, "set_classes"):
                pos_embeddings = model.get_text_pe(self._vocabulary)
                model.set_classes(self._vocabulary, pos_embeddings)

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

        results = model.predict(
            inputs,
            conf=self._conf_threshold,
            imgsz=self._imgsz,
        )
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
        """Convert YOLO + tracker output to the canonical detections payload."""
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
        return {"detections": Detections(items=boxes, class_names=class_names)}

    def get_available_runtimes(self) -> list[str]:
        return ["pytorch", "openvino", "onnx"]

    def supports_training(self, config: dict[str, Any] | None = None) -> bool:
        _ = config
        return False

    def validate_training_dataset(self, dataset, config: dict[str, Any] | None = None) -> list[str]:
        _ = config
        return dataset.validate_for_tool("open_vocab_detector")

    def train(self, model_ref: str, dataset, config: dict[str, Any] | None = None, callbacks=None):
        _ = (model_ref, dataset, config, callbacks)
        raise NotImplementedError("Open-vocabulary YOLO backend does not support training.")

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

    @staticmethod
    def _resolve_checkpoint_path(model_path: str, yolo_settings: Any) -> str:
        """Force bare checkpoint names to resolve under Ultralytics weights_dir."""
        p = Path(model_path)
        if p.suffix.lower() == ".pt" and not p.is_absolute() and p.name == model_path:
            return str(Path(yolo_settings["weights_dir"]) / p.name)
        return model_path
