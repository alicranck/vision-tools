from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vision_tools.core.type_refs import TypeRegistry


class GraphModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FrameInfo(GraphModel):
    frame_idx: int = Field(..., ge=0)
    timestamp: float = Field(0.0, ge=0.0)
    camera_id: str = "default"
    frame_shape: tuple[int, int, int] = (0, 0, 3)
    scene_change_score: float | None = Field(default=None, ge=0.0, le=1.0)


class Image(GraphModel):
    data: Any
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    channels: int = Field(3, gt=0)


class BoundingBox(GraphModel):
    xyxy: list[float] = Field(..., min_length=4, max_length=4)
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    track_id: int | None = None


class Detections(GraphModel):
    items: list[BoundingBox] = Field(default_factory=list)
    class_names: dict[int, str] | None = None


class Track(GraphModel):
    track_id: int
    xyxy: list[float] = Field(..., min_length=4, max_length=4)
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    age: int = Field(0, ge=0)
    hits: int = Field(1, ge=1)


class Tracks(GraphModel):
    items: list[Track] = Field(default_factory=list)


class Crop(GraphModel):
    track_id: int | None = None
    source_index: int = Field(..., ge=0)
    xyxy: list[float] = Field(..., min_length=4, max_length=4)
    image: Any
    class_id: int | None = None
    class_name: str | None = None


class Crops(GraphModel):
    items: list[Crop] = Field(default_factory=list)


class Embedding(GraphModel):
    vector: list[float] = Field(default_factory=list)
    model_id: str
    dimension: int = Field(..., ge=0)


class Caption(GraphModel):
    text: str
    model_id: str


class Classification(GraphModel):
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Classifications(GraphModel):
    items: list[Classification] = Field(default_factory=list)


class Keypoint(GraphModel):
    x: float
    y: float
    confidence: float = Field(..., ge=0.0, le=1.0)


class PoseKeypoints(GraphModel):
    person_id: int
    keypoints: list[Keypoint] = Field(default_factory=list)


class Poses(GraphModel):
    items: list[PoseKeypoints] = Field(default_factory=list)


class SegmentationMask(GraphModel):
    polygon: list[list[float]] = Field(default_factory=list)
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SegmentationMasks(GraphModel):
    items: list[SegmentationMask] = Field(default_factory=list)


class History(GraphModel):
    items: list[Any] = Field(default_factory=list)
    window_start: float | None = Field(default=None, ge=0.0)
    window_end: float | None = Field(default=None, ge=0.0)
    emitted_at_frame: int = Field(..., ge=0)


class Sequence(GraphModel):
    items: list[Any] = Field(default_factory=list)


class Alert(GraphModel):
    severity: Literal["info", "warning", "critical"] = "info"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: float | None = Field(default=None, ge=0.0)
    frame_idx: int | None = Field(default=None, ge=0)


class Alerts(GraphModel):
    items: list[Alert] = Field(default_factory=list)


class MemoryMatch(GraphModel):
    """A single result from a MemoryStore query."""

    entry_id: str
    label: str | None = None
    similarity: float = Field(0.0, ge=0.0, le=1.0)
    media_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryMatches(GraphModel):
    """Collection of results from a MemoryStore query."""

    items: list[MemoryMatch] = Field(default_factory=list)
    store_id: str = ""


for _name, _model in {
    "FrameInfo": FrameInfo,
    "Image": Image,
    "BoundingBox": BoundingBox,
    "Detections": Detections,
    "Track": Track,
    "Tracks": Tracks,
    "Crop": Crop,
    "Crops": Crops,
    "Embedding": Embedding,
    "Caption": Caption,
    "Classification": Classification,
    "Classifications": Classifications,
    "Keypoint": Keypoint,
    "PoseKeypoints": PoseKeypoints,
    "Poses": Poses,
    "SegmentationMask": SegmentationMask,
    "SegmentationMasks": SegmentationMasks,
    "History": History,
    "Sequence": Sequence,
    "Alert": Alert,
    "Alerts": Alerts,
    "MemoryMatch": MemoryMatch,
    "MemoryMatches": MemoryMatches,
}.items():
    TypeRegistry.register_simple(_name, _model)
