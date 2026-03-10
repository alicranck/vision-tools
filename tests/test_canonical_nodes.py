import numpy as np
import pytest

from vision_tools.core.config import NodeConfig, PipelineConfig
from vision_tools.core.graph_types import (
    Alert,
    Alerts,
    BoundingBox,
    Crop,
    Detections,
    FrameInfo,
    History,
    Image,
    Track,
    Tracks,
)
from vision_tools.core.node import NodeContext
from vision_tools.nodes.logic.dynamic_logic_node import DynamicLogicNode
from vision_tools.nodes.state.buffer_node import BufferNode
from vision_tools.nodes.state.track_node import TrackNode
from vision_tools.nodes.utility.crop_node import CropNode
from vision_tools.nodes.utility.filter_node import FilterNode
from vision_tools.pipeline.graph import DAG


def test_filter_node_filters_by_confidence_and_class():
    node = FilterNode("filter", {"min_confidence": 0.5, "allowed_classes": ["person"]})
    detections = Detections(
        items=[
            BoundingBox(xyxy=[0, 0, 10, 10], class_name="person", confidence=0.9),
            BoundingBox(xyxy=[0, 0, 5, 5], class_name="car", confidence=0.9),
            BoundingBox(xyxy=[0, 0, 3, 3], class_name="person", confidence=0.2),
        ]
    )
    result = node.process({"detections": detections}, NodeContext())
    assert len(result["detections"].items) == 1
    assert result["detections"].items[0].class_name == "person"


def test_track_node_assigns_stable_ids():
    node = TrackNode("track", {"iou_threshold": 0.1, "max_age": 5})
    first = node.process(
        {
            "detections": Detections(
                items=[BoundingBox(xyxy=[0, 0, 10, 10], class_name="person", confidence=0.9)]
            )
        },
        NodeContext(),
    )
    second = node.process(
        {
            "detections": Detections(
                items=[BoundingBox(xyxy=[1, 1, 11, 11], class_name="person", confidence=0.8)]
            )
        },
        NodeContext(),
    )
    assert first["tracks"].items[0].track_id == second["tracks"].items[0].track_id


def test_crop_node_crops_tracks():
    node = CropNode("crop", {"padding": 0.0})
    image = Image(data=np.ones((10, 10, 3), dtype=np.uint8), width=10, height=10, channels=3)
    tracks = Tracks(
        items=[Track(track_id=7, xyxy=[1, 2, 5, 6], class_name="person", confidence=0.9)]
    )
    result = node.process({"image": image, "regions": tracks}, NodeContext())
    crop = result["crops"].items[0]
    assert crop.track_id == 7
    assert crop.source_index == 0
    assert crop.image.shape[:2] == (4, 4)


def test_buffer_node_emits_on_stride():
    node = BufferNode(
        "buffer",
        {"item_type": "Tracks", "window_size_seconds": 5.0, "emit_every_frames": 2},
    )
    track_frame = Tracks(items=[Track(track_id=1, xyxy=[0, 0, 1, 1])])
    first = node.process({"items": track_frame}, NodeContext(frame_idx=0, timestamp=0.0))
    second = node.process({"items": track_frame}, NodeContext(frame_idx=1, timestamp=1.0))
    assert first == {}
    assert isinstance(second["history"], History)
    assert len(second["history"].items) == 2


def test_buffer_node_config_validates_item_type():
    node = BufferNode(
        "buffer",
        {"item_type": "Tracks", "window_size_seconds": 5.0, "emit_every_frames": 1},
    )
    dag = DAG(
        PipelineConfig(
            nodes=[
                NodeConfig(node_id="buffer", node_type="buffer", inputs={"items": "det.detections"}),
                NodeConfig(node_id="det", node_type="filter", inputs={"detections": "input.image"}, config={"min_confidence": 0.5}),
            ]
        )
    )
    with pytest.raises(Exception):
        node.validate_config(dag, {"det": FilterNode("det", {"min_confidence": 0.5})})


def test_dynamic_logic_requires_declared_outputs():
    code = """
def execute(inputs, context, config, state):
    return {"alerts": {"items": [{"message": "ok"}]}}
"""
    node = DynamicLogicNode(
        "logic",
        {
            "input_ports": {"history": "History[Tracks]"},
            "output_ports": {"alerts": "Alerts"},
            "code": code,
        },
    )
    history = History(items=[Tracks(items=[])], window_start=0.0, window_end=0.0, emitted_at_frame=0)
    result = node.process({"history": history}, NodeContext())
    assert result["alerts"].items[0].message == "ok"


def test_dynamic_logic_state_persists():
    code = """
def execute(inputs, context, config, state):
    state["count"] = state.get("count", 0) + 1
    return {"alerts": {"items": [{"message": str(state["count"])}]}}
"""
    node = DynamicLogicNode(
        "logic",
        {
            "input_ports": {"history": "History[Tracks]"},
            "output_ports": {"alerts": "Alerts"},
            "code": code,
        },
    )
    history = History(items=[], window_start=None, window_end=None, emitted_at_frame=0)
    first = node.process({"history": history}, NodeContext())
    second = node.process({"history": history}, NodeContext())
    assert first["alerts"].items[0].message == "1"
    assert second["alerts"].items[0].message == "2"
