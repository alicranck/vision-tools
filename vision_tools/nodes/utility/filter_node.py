from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from vision_tools.core.graph_types import Detections
from vision_tools.nodes.logic.logic_node import LogicNode


class FilterNodeConfig(BaseModel):
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Discard detections below this confidence score. Leave unset to skip confidence filtering.", json_schema_extra={"x-ui-widget": "slider"})
    allowed_classes: list[str] = Field(default_factory=list, description="Keep only detections whose class name is in this list. Leave empty to allow all classes.")
    allowed_class_ids: list[int] = Field(default_factory=list, description="Keep only detections whose class ID is in this list. Leave empty to allow all class IDs.", json_schema_extra={"x-advanced": True})

    @model_validator(mode="after")
    def validate_non_empty(self) -> "FilterNodeConfig":
        if (
            self.min_confidence is None
            and not self.allowed_classes
            and not self.allowed_class_ids
        ):
            raise ValueError("At least one filtering criterion must be supplied.")
        return self


class FilterNode(LogicNode):
    InputPorts = {"detections": "Detections"}
    OutputPorts = {"detections": "Detections"}

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = FilterNodeConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._config = validated

    def execute(self, inputs, context):
        detections: Detections = inputs["detections"]
        filtered = []
        for item in detections.items:
            if (
                self._config.min_confidence is not None
                and (item.confidence is None or item.confidence < self._config.min_confidence)
            ):
                continue
            if self._config.allowed_classes and item.class_name not in self._config.allowed_classes:
                continue
            if self._config.allowed_class_ids and item.class_id not in self._config.allowed_class_ids:
                continue
            filtered.append(item)
        return {
            "detections": Detections(items=filtered, class_names=detections.class_names)
        }

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return FilterNodeConfig
