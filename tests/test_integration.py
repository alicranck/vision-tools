import numpy as np
import pytest

from vision_tools.core.config import NodeConfig, PipelineConfig
from vision_tools.core.graph_types import BoundingBox, Detections
from vision_tools.core.node import Node, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.pipeline import Pipeline


class IntegrationDetector(Node):
    InputPorts = {"image": "Image"}
    OutputPorts = {"detections": "Detections"}

    def __init__(self, node_id="det", config=None):
        super().__init__(node_id, config or {})
        self._state = NodeState.READY

    def process(self, inputs, context):
        return {
            "detections": Detections(
                items=[
                    BoundingBox(xyxy=[0, 0, 100, 100], class_name="person", confidence=0.95),
                    BoundingBox(xyxy=[50, 50, 150, 150], class_name="car", confidence=0.3),
                ],
                class_names={0: "person", 1: "car"},
            )
        }


@pytest.fixture(autouse=True)
def register_integration_detector():
    saved = dict(NodeRegistry._registry)
    saved_categories = dict(NodeRegistry._categories)
    NodeRegistry._registry["integration_detector"] = IntegrationDetector
    NodeRegistry._categories["integration_detector"] = "testing"
    yield
    NodeRegistry._registry.clear()
    NodeRegistry._registry.update(saved)
    NodeRegistry._categories.clear()
    NodeRegistry._categories.update(saved_categories)


def test_detector_filter_track_pipeline():
    config = PipelineConfig(
        name="people_only",
        nodes=[
            NodeConfig(
                node_id="det",
                node_type="integration_detector",
                inputs={"image": "input.image"},
            ),
            NodeConfig(
                node_id="filter",
                node_type="filter",
                config={"min_confidence": 0.5, "allowed_classes": ["person"]},
                inputs={"detections": "det.detections"},
            ),
            NodeConfig(
                node_id="track",
                node_type="track",
                config={"iou_threshold": 0.2},
                inputs={"detections": "filter.detections"},
            ),
        ],
        outputs={"tracks": "track.tracks"},
    )
    pipeline = Pipeline(config)
    result = pipeline.run(np.zeros((128, 128, 3), dtype=np.uint8))
    assert len(result["tracks"].items) == 1
    assert result["tracks"].items[0].class_name == "person"


def test_detector_track_buffer_dynamic_logic_pipeline():
    code = """
def execute(inputs, context, config, state):
    track_count = sum(len(batch.items) for batch in inputs["history"].items)
    return {"alerts": {"items": [{"message": f"tracks={track_count}"}]}}
"""
    config = PipelineConfig(
        name="buffered",
        nodes=[
            NodeConfig(
                node_id="det",
                node_type="integration_detector",
                inputs={"image": "input.image"},
            ),
            NodeConfig(
                node_id="track",
                node_type="track",
                inputs={"detections": "det.detections"},
            ),
            NodeConfig(
                node_id="buffer",
                node_type="buffer",
                config={"item_type": "Tracks", "window_size_seconds": 10.0, "emit_every_frames": 1},
                inputs={"items": "track.tracks"},
            ),
            NodeConfig(
                node_id="logic",
                node_type="dynamic_logic",
                config={
                    "input_ports": {"history": "History[Tracks]"},
                    "output_ports": {"alerts": "Alerts"},
                    "code": code,
                },
                inputs={"history": "buffer.history"},
            ),
        ],
        outputs={"alerts": "logic.alerts"},
    )
    pipeline = Pipeline(config)
    result = pipeline.run(np.zeros((64, 64, 3), dtype=np.uint8))
    assert result["alerts"].items[0].message == "tracks=2"
