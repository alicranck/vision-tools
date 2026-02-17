"""
CropTool — Deterministic crop around a bounding box.

Takes an image and a BoundingBox, applies optional transformations
(padding, scaling, shifting), and returns the cropped image plus
the adjusted BoundingBox in the original coordinate frame.

This is a non-model tool: it has no ML Model, no warmup, no state machine.
It participates in the pipeline DAG as a regular node.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ...utils.schemas import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class CropConfig:
    """Configuration for a crop operation."""
    pad: int = 0            # Absolute pixel padding around the bbox
    scale: float = 1.0      # Scale factor around bbox center (1.0 = no scale)
    x_shift: int = 0        # Horizontal shift in pixels
    y_shift: int = 0        # Vertical shift in pixels


@dataclass
class CropResult:
    """Result of a crop operation."""
    cropped_image: np.ndarray       # The cropped region as an image
    original_bbox: BoundingBox      # The original input bbox
    adjusted_bbox: BoundingBox      # The bbox after pad/scale/shift (in original coords)
    crop_offset: Tuple[int, int]    # (x_offset, y_offset) of crop origin in original image


class CropTool:
    """
    Crops an image region around a bounding box.

    Not a BaseVisionTool subclass — this is a pure utility node
    with no model, no state machine, no warmup.

    Usage:
        crop_tool = CropTool(pad=20, scale=1.2)
        result = crop_tool.crop(image, bbox)
        # result.cropped_image, result.adjusted_bbox, result.crop_offset
    """

    def __init__(self, pad: int = 0, scale: float = 1.0,
                 x_shift: int = 0, y_shift: int = 0):
        self.config = CropConfig(
            pad=pad,
            scale=scale,
            x_shift=x_shift,
            y_shift=y_shift,
        )

    @classmethod
    def from_config(cls, config: dict) -> "CropTool":
        """Create from a config dict (pipeline integration)."""
        return cls(
            pad=config.get("pad", 0),
            scale=config.get("scale", 1.0),
            x_shift=config.get("x_shift", 0),
            y_shift=config.get("y_shift", 0),
        )

    def crop(self, image: np.ndarray, bbox: BoundingBox) -> CropResult:
        """
        Crop the image around the bounding box.

        Args:
            image: HxWxC numpy array
            bbox: BoundingBox with xyxy coordinates

        Returns:
            CropResult with cropped image, adjusted bbox, and crop offset
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.xyxy

        # Apply scale around center
        if self.config.scale != 1.0:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = (x2 - x1) * self.config.scale
            bh = (y2 - y1) * self.config.scale
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

        # Apply shift
        x1 += self.config.x_shift
        x2 += self.config.x_shift
        y1 += self.config.y_shift
        y2 += self.config.y_shift

        # Apply padding
        x1 -= self.config.pad
        y1 -= self.config.pad
        x2 += self.config.pad
        y2 += self.config.pad

        # Clamp to image bounds
        crop_x1 = max(0, int(x1))
        crop_y1 = max(0, int(y1))
        crop_x2 = min(w, int(x2))
        crop_y2 = min(h, int(y2))

        # Guard against degenerate crops
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            logger.warning(
                f"Degenerate crop region: ({crop_x1},{crop_y1})-({crop_x2},{crop_y2}). "
                f"Returning 1x1 patch."
            )
            crop_x2 = max(crop_x1 + 1, min(w, crop_x1 + 1))
            crop_y2 = max(crop_y1 + 1, min(h, crop_y1 + 1))

        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        adjusted_bbox = BoundingBox(
            xyxy=[float(crop_x1), float(crop_y1), float(crop_x2), float(crop_y2)],
            class_id=bbox.class_id,
            confidence=bbox.confidence,
            class_name=bbox.class_name,
            tracker_id=bbox.tracker_id,
        )

        return CropResult(
            cropped_image=cropped,
            original_bbox=bbox,
            adjusted_bbox=adjusted_bbox,
            crop_offset=(crop_x1, crop_y1),
        )

    def crop_best(self, image: np.ndarray, boxes: list,
                  class_name: Optional[str] = None) -> Optional[CropResult]:
        """
        Crop around the highest-confidence box, optionally filtering by class.

        Args:
            image: HxWxC numpy array
            boxes: List of BoundingBox dicts (from DetectionResult.model_dump())
            class_name: Optional class filter

        Returns:
            CropResult for the best matching box, or None if no match
        """
        # Convert dicts to BoundingBox if needed
        parsed_boxes = []
        for b in boxes:
            if isinstance(b, dict):
                parsed_boxes.append(BoundingBox(**b))
            else:
                parsed_boxes.append(b)

        # Filter by class
        if class_name is not None:
            parsed_boxes = [b for b in parsed_boxes if b.class_name == class_name]

        if not parsed_boxes:
            logger.debug(f"No boxes found for class '{class_name}'")
            return None

        # Pick highest confidence
        best = max(parsed_boxes, key=lambda b: b.confidence)
        return self.crop(image, best)
