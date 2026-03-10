from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from vision_tools.core.graph_types import Crop, Crops, Detections, Image, Tracks
from vision_tools.core.registry import NodeRegistry
from vision_tools.nodes.logic.logic_node import LogicNode


class CropNodeConfig(BaseModel):
    padding: float = Field(0.0, ge=0.0)
    clip_to_image: bool = True
    skip_invalid_boxes: bool = True


@NodeRegistry.register("crop", category="canonical")
class CropNode(LogicNode):
    InputPorts = {"image": "Image", "regions": "Detections | Tracks"}
    OutputPorts = {"crops": "Crops"}

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = CropNodeConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._config = validated

    def execute(self, inputs: dict[str, Any], context):
        image: Image = inputs["image"]
        regions = inputs["regions"]
        raw_image = image.data
        width = image.width
        height = image.height
        items = regions.items
        crops = []

        for index, item in enumerate(items):
            x1, y1, x2, y2 = map(float, item.xyxy)
            pad_x = (x2 - x1) * self._config.padding
            pad_y = (y2 - y1) * self._config.padding
            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

            if self._config.clip_to_image:
                x1 = max(0.0, min(float(width), x1))
                y1 = max(0.0, min(float(height), y1))
                x2 = max(0.0, min(float(width), x2))
                y2 = max(0.0, min(float(height), y2))

            ix1, iy1, ix2, iy2 = map(int, (round(x1), round(y1), round(x2), round(y2)))
            if ix2 <= ix1 or iy2 <= iy1:
                if self._config.skip_invalid_boxes:
                    continue
                raise ValueError(f"{self.node_id}: invalid crop bounds {(ix1, iy1, ix2, iy2)}")

            crop_image = raw_image[iy1:iy2, ix1:ix2]
            crops.append(
                Crop(
                    track_id=getattr(item, "track_id", None),
                    source_index=index,
                    xyxy=[float(ix1), float(iy1), float(ix2), float(iy2)],
                    image=crop_image,
                    class_id=getattr(item, "class_id", None),
                    class_name=getattr(item, "class_name", None),
                )
            )

        return {"crops": Crops(items=crops)}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return CropNodeConfig
