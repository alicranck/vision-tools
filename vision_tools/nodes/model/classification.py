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
from vision_tools.core.graph_types import Classifications
from vision_tools.nodes.model.model_node import ModelNode
from vision_tools.runtime.model_catalog import ModelCatalog


class ClassifierConfig(BaseModel):
    task: InferenceTask = Field(default=InferenceTask.CLASSIFICATION)
    model_family: str = Field("yolo_cls")
    size: ModelSize = Field(ModelSize.SMALL)
    device: DeviceTarget = Field(DeviceTarget.AUTO)
    imgsz: int = Field(224, gt=0)
    topk: int = Field(3, ge=1, le=20)
    training_mode: TrainingMode = Field(TrainingMode.NONE)
    dataset_id: str | None = None
    model_source: ModelSource = Field(ModelSource.BASE)
    model_asset_version_id: str | None = None
    artifact_path: str | None = None


class Classifier(ModelNode):
    InputPorts = {"image": "Image"}
    OutputPorts = {"classifications": "Classifications"}
    TrainingMetadata: ClassVar[dict[str, Any]] = {
        "supported": True,
        "task": "classification",
        "annotation_type": "classification",
    }

    def __init__(self, node_id: str = "classifier", config: dict | None = None, **kwargs) -> None:
        validated = ClassifierConfig.model_validate(config or {})
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
        if not isinstance(outputs, dict):
            raise TypeError(f"{self.node_id}: backend output must be a dict.")
        if "classifications" not in outputs:
            raise TypeError(f"{self.node_id}: backend output must define 'classifications'.")
        classifications = outputs["classifications"]
        if hasattr(classifications, "model_dump"):
            classifications = classifications.model_dump()
        return {"classifications": Classifications.model_validate(classifications)}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return ClassifierConfig

    @classmethod
    def get_config_options(cls) -> dict[str, Any]:
        return {"model_catalog": ModelCatalog.list_options(task=InferenceTask.CLASSIFICATION)}

    @classmethod
    def get_training_metadata(cls) -> dict[str, Any] | None:
        return dict(cls.TrainingMetadata)
