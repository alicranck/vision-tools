from __future__ import annotations

from pydantic import BaseModel, Field

from vision_tools.core.graph_types import Detections, Track, Tracks
from vision_tools.nodes.logic.logic_node import LogicNode


class TrackNodeConfig(BaseModel):
    method: str = Field("iou", pattern=r"^iou$", description="Matching algorithm used to associate detections to existing tracks.", json_schema_extra={"x-advanced": True})
    iou_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum IoU overlap required to match a detection to an existing track.", json_schema_extra={"x-ui-widget": "slider"})
    max_age: int = Field(30, ge=0, description="Number of consecutive frames a track can go unmatched before it is dropped.")
    min_hits: int = Field(1, ge=1, description="Minimum number of consecutive matched frames before a track is considered confirmed and emitted.")


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter_area
    return inter_area / denom if denom > 0 else 0.0


class TrackNode(LogicNode):
    InputPorts = {"detections": "Detections"}
    OutputPorts = {"tracks": "Tracks"}

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        validated = TrackNodeConfig.model_validate(config or {})
        super().__init__(node_id, validated.model_dump())
        self._config = validated
        self._next_track_id = 1
        self._active_tracks: dict[int, Track] = {}

    def execute(self, inputs, context):
        detections: Detections = inputs["detections"]
        unmatched_tracks = set(self._active_tracks.keys())
        updated_tracks: dict[int, Track] = {}

        for detection in detections.items:
            best_track_id = None
            best_iou = 0.0
            for track_id in list(unmatched_tracks):
                current = self._active_tracks[track_id]
                score = _iou(current.xyxy, detection.xyxy)
                if score > best_iou and score >= self._config.iou_threshold:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                updated_tracks[track_id] = Track(
                    track_id=track_id,
                    xyxy=list(detection.xyxy),
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    age=0,
                    hits=1,
                )
                continue

            unmatched_tracks.remove(best_track_id)
            current = self._active_tracks[best_track_id]
            updated_tracks[best_track_id] = Track(
                track_id=best_track_id,
                xyxy=list(detection.xyxy),
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                age=0,
                hits=current.hits + 1,
            )

        for track_id in unmatched_tracks:
            current = self._active_tracks[track_id]
            aged = current.model_copy(update={"age": current.age + 1})
            if aged.age <= self._config.max_age:
                updated_tracks[track_id] = aged

        self._active_tracks = updated_tracks
        visible = [
            track
            for track in self._active_tracks.values()
            if track.hits >= self._config.min_hits
        ]
        return {"tracks": Tracks(items=visible)}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return TrackNodeConfig
