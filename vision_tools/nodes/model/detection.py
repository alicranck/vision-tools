from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.config import (
    DeviceTarget,
    InferenceTask,
    ModelIntent,
    ModelSize,
    ModelSource,
    TrainingMode,
)
from vision_tools.core.graph_types import BoundingBox, Detections
from vision_tools.nodes.model.model_node import ModelNode
from vision_tools.runtime.model_catalog import ModelCatalog


class DetectionNodeConfig(BaseModel):
    task: InferenceTask = Field(default=InferenceTask.DETECTION)
    model_family: str = Field("yolo_detector")
    size: ModelSize = Field(ModelSize.SMALL)
    device: DeviceTarget = Field(DeviceTarget.AUTO)
    vocabulary: list[str] = Field(default_factory=list)
    imgsz: int = Field(640, gt=0)
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    prompt_free: bool = True
    training_mode: TrainingMode = Field(TrainingMode.NONE)
    dataset_id: str | None = None
    model_source: ModelSource = Field(ModelSource.BASE)
    model_asset_version_id: str | None = None
    artifact_path: str | None = None


class _BaseDetector(ModelNode):
    InputPorts = {"image": "Image"}
    OutputPorts = {"detections": "Detections"}
    TrainingMetadata: ClassVar[dict[str, Any]] = {"supported": False, "task": "detection"}

    def __init__(self, node_id: str, config: dict | None = None, **kwargs) -> None:
        validated = DetectionNodeConfig.model_validate(config or {})
        intent = ModelIntent(
            task=validated.task,
            model_family=validated.model_family,
            size=validated.size,
            device=validated.device,
        )
        resolved = ModelCatalog.resolve(intent)

        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(
                task=resolved.backend_task,
                model=resolved.backend_model,
            )

        resolved_config = validated.model_dump(mode="json")
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

    def preprocess(self, inputs: dict[str, Any], context) -> Any:
        _ = context
        return inputs["image"].data

    def normalize_outputs(self, outputs: Any) -> dict[str, Any]:
        if isinstance(outputs, Detections):
            return {"detections": outputs}

        if isinstance(outputs, dict) and "detections" in outputs:
            return outputs

        if not isinstance(outputs, dict):
            raise TypeError(f"{self.node_id}: backend output must be a dict.")

        boxes = outputs.get("boxes", [])
        class_names = outputs.get("class_names")
        normalized_boxes = []
        for box in boxes:
            if hasattr(box, "model_dump"):
                box = box.model_dump()
            normalized_boxes.append(
                BoundingBox(
                    xyxy=list(map(float, box["xyxy"])),
                    class_id=box.get("class_id"),
                    class_name=box.get("class_name"),
                    confidence=box.get("confidence"),
                    track_id=box.get("track_id", box.get("tracker_id")),
                )
            )
        return {"detections": Detections(items=normalized_boxes, class_names=class_names)}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return DetectionNodeConfig

    @classmethod
    def get_config_options(cls) -> dict[str, Any]:
        return {"model_catalog": ModelCatalog.list_options()}

    @classmethod
    def get_training_metadata(cls) -> dict[str, Any] | None:
        return dict(cls.TrainingMetadata)


class OpenVocabularyDetector(_BaseDetector):
    TrainingMetadata = {
        "supported": False,
        "task": "open_vocab_detection",
        "annotation_type": "bounding_box",
    }

    def __init__(
        self,
        node_id: str = "open_vocab_detector",
        config: dict | None = None,
        **kwargs,
    ) -> None:
        merged = {"task": InferenceTask.OPEN_VOCAB_DETECTION, "prompt_free": False, **(config or {})}
        super().__init__(node_id=node_id, config=merged, **kwargs)


class Detector(_BaseDetector):
    TrainingMetadata = {
        "supported": True,
        "task": "detection",
        "annotation_type": "bounding_box",
    }

    def __init__(self, node_id: str = "detector", config: dict | None = None, **kwargs) -> None:
        merged = {"task": InferenceTask.DETECTION, "prompt_free": True, **(config or {})}
        super().__init__(node_id=node_id, config=merged, **kwargs)


# Backward compatibility for existing imports/tests.
ObjectDetector = OpenVocabularyDetector
