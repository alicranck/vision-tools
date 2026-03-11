"""
Unified annotation model for vision-tools datasets.

Polygon is the canonical representation. Bounding boxes and masks are
derived views — a bbox is the bounding rectangle of the polygon, and a
mask is the rasterised polygon.

All coordinates are normalised to [0, 1] relative to image dimensions.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field


class Annotation(BaseModel):
    """A single annotation on a frame. Polygon is the source of truth."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    class_name: str
    class_id: Optional[int] = None  # assigned during training export
    polygon: List[Tuple[float, float]] = Field(
        ..., min_length=3, description="Normalised (x, y) vertices, 0-1"
    )
    confidence: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- derived views -------------------------------------------------------

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding rect as (x, y, w, h) from top-left, normalised."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        x_min, y_min = min(xs), min(ys)
        return (x_min, y_min, max(xs) - x_min, max(ys) - y_min)

    @property
    def is_rectangle(self) -> bool:
        """True when the polygon is a 4-vertex axis-aligned rectangle."""
        if len(self.polygon) != 4:
            return False
        xs = sorted(set(p[0] for p in self.polygon))
        ys = sorted(set(p[1] for p in self.polygon))
        return len(xs) == 2 and len(ys) == 2

    def to_mask(self, width: int, height: int) -> np.ndarray:
        """Rasterise polygon to a binary mask of shape (height, width)."""
        import cv2

        pts = np.array(
            [(int(x * width), int(y * height)) for x, y in self.polygon],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    # -- factories -----------------------------------------------------------

    @classmethod
    def from_bbox(
        cls,
        x: float,
        y: float,
        w: float,
        h: float,
        class_name: str,
        **kwargs: Any,
    ) -> Annotation:
        """Create a rectangular polygon from (x, y, w, h) normalised coords."""
        polygon = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h),
        ]
        return cls(class_name=class_name, polygon=polygon, **kwargs)

    # -- YOLO conversion (internal) ------------------------------------------

    def to_yolo_bbox_line(self, class_id: int) -> str:
        """YOLO detection label: ``class_id cx cy w h`` (normalised)."""
        x, y, w, h = self.bbox
        cx = x + w / 2
        cy = y + h / 2
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def to_yolo_segment_line(self, class_id: int) -> str:
        """YOLO segmentation label: ``class_id x1 y1 x2 y2 ...`` (normalised)."""
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.polygon)
        return f"{class_id} {coords}"


class AnnotatedFrame(BaseModel):
    """All annotations for a single frame / dataset entry."""

    annotations: List[Annotation] = Field(default_factory=list)
    image_label: Optional[str] = Field(
        default=None,
        description="Single image-level classification label.",
    )
    labels: List[str] = Field(
        default_factory=list,
        description="Generic image-level labels/tags for non-classification use cases.",
    )

    @property
    def class_names(self) -> list[str]:
        """Unique class names across all annotations, sorted."""
        names = {a.class_name for a in self.annotations} | set(self.labels)
        if self.image_label:
            names.add(self.image_label)
        return sorted(names)

    @property
    def is_empty(self) -> bool:
        return len(self.annotations) == 0 and len(self.labels) == 0 and self.image_label is None
