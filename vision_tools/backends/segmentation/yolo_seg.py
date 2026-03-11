from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.graph_types import SegmentationMask, SegmentationMasks
from vision_tools.core.node import NodeContext
from vision_tools.training.artifacts import TrainedArtifact


@BackendRegistry.register(task="segmentation", model="yolo_seg")
class YoloSegmentationBackend:
    def __init__(self) -> None:
        self._imgsz = 640
        self._conf_threshold = 0.25

    def configure(self, config: dict[str, Any]) -> None:
        self._imgsz = int(config.get("imgsz", self._imgsz))
        self._conf_threshold = float(config.get("conf_threshold", self._conf_threshold))

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        from ultralytics import YOLO

        _ = device
        return YOLO(model_path)

    def infer(self, model: Any, inputs: Any) -> Any:
        results = model.predict(inputs, imgsz=self._imgsz, conf=self._conf_threshold)
        return results[0]

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        _ = context
        names = raw_output.names or {}
        masks = []
        if getattr(raw_output, "masks", None) is not None and raw_output.boxes is not None:
            polygons = getattr(raw_output.masks, "xy", []) or []
            for index, polygon in enumerate(polygons):
                box = raw_output.boxes[index]
                class_id = int(box.cls[0].item()) if box.cls is not None else -1
                confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
                masks.append(
                    SegmentationMask(
                        polygon=[[float(x), float(y)] for x, y in polygon.tolist()],
                        class_id=class_id,
                        class_name=names.get(class_id),
                        confidence=confidence,
                    )
                )
        return {"segmentation_masks": SegmentationMasks(items=masks)}

    def get_available_runtimes(self) -> list[str]:
        return ["pytorch", "cpu", "auto"]

    def supports_training(self, config: dict[str, Any] | None = None) -> bool:
        _ = config
        return True

    def validate_training_dataset(self, dataset, config: dict[str, Any] | None = None) -> list[str]:
        _ = config
        errors = dataset.validate_for_tool("segmenter")
        polygon_count = dataset.polygon_annotation_count()
        if polygon_count == 0:
            errors.append("Segmentation training requires polygon annotations.")
        return errors

    def train(self, model_ref: str, dataset, config: dict[str, Any] | None = None, callbacks=None) -> TrainedArtifact:
        from ultralytics import YOLO

        _ = callbacks
        train_config = dict(config or {})
        data_ref = dataset.materialize_for_task("segmentation")
        model = YOLO(model_ref)
        results = model.train(data=data_ref, task="segment", **train_config)
        save_dir = Path(getattr(results, "save_dir", ""))
        artifact_path = save_dir / "weights" / "best.pt"
        if not artifact_path.exists():
            artifact_path = Path(model_ref)
        return TrainedArtifact(
            artifact_path=str(artifact_path),
            backend_task="segmentation",
            backend_model="yolo_seg",
            task="segmentation",
            model_family="yolo_seg",
            base_checkpoint_id=model_ref,
            metadata={"train_config": train_config, "data_ref": data_ref},
        )
