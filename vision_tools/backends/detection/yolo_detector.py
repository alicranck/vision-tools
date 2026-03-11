from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.graph_types import BoundingBox, Detections
from vision_tools.core.node import NodeContext
from vision_tools.training.artifacts import TrainedArtifact


@BackendRegistry.register(task="detection", model="yolo_detector")
class YoloDetectorBackend:
    def __init__(self) -> None:
        self._imgsz = 640
        self._conf_threshold = 0.25

    def configure(self, config: dict[str, Any]) -> None:
        self._imgsz = int(config.get("imgsz", self._imgsz))
        self._conf_threshold = float(config.get("conf_threshold", self._conf_threshold))

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        from ultralytics import YOLO

        model = YOLO(model_path)
        _ = device
        return model

    def infer(self, model: Any, inputs: Any) -> Any:
        results = model.predict(inputs, conf=self._conf_threshold, imgsz=self._imgsz)
        return results[0]

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        _ = context
        names = raw_output.names or {}
        boxes = []
        if raw_output.boxes is not None:
            for box in raw_output.boxes:
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                class_id = int(box.cls[0].item()) if box.cls is not None else -1
                confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
                boxes.append(
                    BoundingBox(
                        xyxy=xyxy,
                        class_id=class_id,
                        class_name=names.get(class_id),
                        confidence=confidence,
                    )
                )
        return {"detections": Detections(items=boxes, class_names=names)}

    def get_available_runtimes(self) -> list[str]:
        return ["pytorch", "cpu", "auto"]

    def supports_training(self, config: dict[str, Any] | None = None) -> bool:
        _ = config
        return True

    def validate_training_dataset(self, dataset, config: dict[str, Any] | None = None) -> list[str]:
        _ = config
        return dataset.validate_for_tool("detector")

    def train(
        self,
        model_ref: str,
        dataset,
        config: dict[str, Any] | None = None,
        callbacks=None,
    ) -> TrainedArtifact:
        from ultralytics import YOLO

        train_config = dict(config or {})
        data_ref = dataset.materialize_for_task("detection")
        model = YOLO(model_ref)
        results = model.train(data=data_ref, **train_config)
        save_dir = Path(getattr(results, "save_dir", ""))
        artifact_path = save_dir / "weights" / "best.pt"
        if not artifact_path.exists():
            artifact_path = Path(model_ref)
        return TrainedArtifact(
            artifact_path=str(artifact_path),
            backend_task="detection",
            backend_model="yolo_detector",
            task="detection",
            model_family="yolo_detector",
            base_checkpoint_id=model_ref,
            metrics={},
            metadata={"train_config": train_config, "data_ref": data_ref},
        )
