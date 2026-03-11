from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.graph_types import Classification, Classifications
from vision_tools.core.node import NodeContext
from vision_tools.training.artifacts import TrainedArtifact


@BackendRegistry.register(task="classification", model="yolo_cls")
class YoloClassificationBackend:
    def __init__(self) -> None:
        self._imgsz = 224
        self._topk = 3

    def configure(self, config: dict[str, Any]) -> None:
        self._imgsz = int(config.get("imgsz", self._imgsz))
        self._topk = int(config.get("topk", self._topk))

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        from ultralytics import YOLO

        _ = device
        return YOLO(model_path)

    def infer(self, model: Any, inputs: Any) -> Any:
        results = model.predict(inputs, imgsz=self._imgsz)
        return results[0]

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        _ = context
        probs = getattr(raw_output, "probs", None)
        names = getattr(raw_output, "names", {}) or {}
        items = []
        if probs is not None:
            top_indices = probs.top5[: self._topk] if hasattr(probs, "top5") else []
            for index in top_indices:
                confidence = float(probs.data[index].item()) if hasattr(probs, "data") else None
                items.append(
                    Classification(
                        class_id=int(index),
                        class_name=names.get(int(index)),
                        confidence=confidence,
                    )
                )
        return {"classifications": Classifications(items=items)}

    def get_available_runtimes(self) -> list[str]:
        return ["pytorch", "cpu", "auto"]

    def supports_training(self, config: dict[str, Any] | None = None) -> bool:
        _ = config
        return True

    def validate_training_dataset(self, dataset, config: dict[str, Any] | None = None) -> list[str]:
        _ = config
        errors = []
        if dataset.num_images == 0:
            errors.append("Dataset contains no images")
        try:
            dataset.materialize_for_task("classification")
        except Exception as exc:
            errors.append(str(exc))
        return errors

    def train(self, model_ref: str, dataset, config: dict[str, Any] | None = None, callbacks=None) -> TrainedArtifact:
        from ultralytics import YOLO

        _ = callbacks
        train_config = dict(config or {})
        data_ref = dataset.materialize_for_task("classification")
        model = YOLO(model_ref)
        results = model.train(data=data_ref, task="classify", **train_config)
        save_dir = Path(getattr(results, "save_dir", ""))
        artifact_path = save_dir / "weights" / "best.pt"
        if not artifact_path.exists():
            artifact_path = Path(model_ref)
        return TrainedArtifact(
            artifact_path=str(artifact_path),
            backend_task="classification",
            backend_model="yolo_cls",
            task="classification",
            model_family="yolo_cls",
            base_checkpoint_id=model_ref,
            metadata={"train_config": train_config, "data_ref": data_ref},
        )
