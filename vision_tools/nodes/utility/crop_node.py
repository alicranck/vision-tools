from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from vision_tools.core.graph_types import Detections, Image, Sequence, Tracks
from vision_tools.nodes.logic.logic_node import LogicNode


class CropNodeConfig(BaseModel):
    padding: float = Field(0.0, ge=0.0, description="Fractional padding added around each bounding box before cropping (e.g. 0.1 = 10% of box size).")
    clip_to_image: bool = Field(True, description="Clamp crop coordinates to the image boundary so crops never exceed the frame edges.")
    skip_invalid_boxes: bool = Field(True, description="Skip boxes whose crop area is zero instead of raising an error.", json_schema_extra={"x-advanced": True})


class CropNode(LogicNode):
    InputPorts = {"image": "Image", "regions": "Detections | Tracks"}
    OutputPorts = {"images": "Sequence[Image]", "regions": "Tracks | Detections"}

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = CropNodeConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._config = validated

    def execute(self, inputs: dict[str, Any], context):
        image: Image = inputs["image"]
        regions = inputs["regions"]
        raw_image = np.asarray(image.data)
        width = image.width
        height = image.height

        images: list[Image] = []
        kept_items: list[Any] = []

        for item in regions.items:
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

            crop_arr = raw_image[iy1:iy2, ix1:ix2]
            h, w = crop_arr.shape[:2]
            ch = int(crop_arr.shape[2]) if crop_arr.ndim == 3 else 1
            images.append(Image(data=crop_arr, width=w, height=h, channels=ch))
            kept_items.append(item)

        if isinstance(regions, Tracks):
            filtered_regions = Tracks(items=kept_items)
        else:
            filtered_regions = Detections(
                items=kept_items,
                class_names=getattr(regions, "class_names", None),
            )

        return {"images": Sequence(items=images), "regions": filtered_regions}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return CropNodeConfig
