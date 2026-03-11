"""
RTDetrBackend — RT-DETR model loading, inference, postprocessing.
"""
from __future__ import annotations

from typing import Any

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.graph_types import BoundingBox, Detections
from vision_tools.core.node import NodeContext


@BackendRegistry.register(task="detection", model="rtdetr")
class RTDetrBackend:
    """Backend for RT-DETR object detection."""

    def __init__(self) -> None:
        self._imgsz = 640
        self._conf_threshold = 0.25
        self._vocabulary: list[str] = []

    def configure(self, config: dict[str, Any]) -> None:
        self._imgsz = int(config.get("imgsz", self._imgsz))
        self._conf_threshold = float(config.get("conf_threshold", self._conf_threshold))
        self._vocabulary = list(config.get("vocabulary", self._vocabulary) or [])

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        from ultralytics import RTDETR

        # RT-DETR runtime selection is handled by underlying ultralytics stack.
        _ = device
        return RTDETR(model_path)

    def infer(self, model: Any, inputs: Any) -> Any:
        results = model.predict(inputs, conf=self._conf_threshold, imgsz=self._imgsz)
        return results[0]

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        _ = context
        boxes_out: list[BoundingBox] = []
        names = raw_output.names or {}

        if raw_output.boxes is not None:
            for box in raw_output.boxes:
                xyxy = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item()) if box.cls is not None else -1
                confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
                boxes_out.append(
                    BoundingBox(
                        xyxy=[float(v) for v in xyxy],
                        class_id=class_id,
                        confidence=confidence,
                        class_name=names.get(class_id),
                    )
                )

        return {"detections": Detections(items=boxes_out, class_names=names)}

    def get_available_runtimes(self) -> list[str]:
        return ["pytorch", "cpu", "auto"]

    def supports_training(self, config: dict[str, Any] | None = None) -> bool:
        _ = config
        return False

    def validate_training_dataset(self, dataset, config: dict[str, Any] | None = None) -> list[str]:
        _ = config
        return dataset.validate_for_tool("detector")

    def train(self, model_ref: str, dataset, config: dict[str, Any] | None = None, callbacks=None):
        _ = (model_ref, dataset, config, callbacks)
        raise NotImplementedError("RT-DETR backend does not support training.")
