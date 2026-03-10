import numpy as np
import pytest

from vision_tools.core.config import NodeConfig, PipelineConfig
from vision_tools.core.graph_types import (
    Alert,
    Alerts,
    BoundingBox,
    Detections,
    FrameInfo,
    Image,
)
from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.pipeline import Pipeline
from vision_tools.pipeline.executor import PipelineExecutor
from vision_tools.pipeline.graph import DAG, CycleError
from vision_tools.pipeline.validator import SchemaValidationError, SchemaValidator


class StubDetectorNode(Node):
    InputPorts = {"image": "Image"}
    OutputPorts = {"detections": "Detections"}

    def __init__(self, node_id="det", config=None):
        super().__init__(node_id, config or {})
        self._state = NodeState.READY

    def process(self, inputs, context):
        return {
            "detections": Detections(
                items=[
                    BoundingBox(
                        xyxy=[0, 0, 10, 10],
                        class_id=0,
                        class_name="person",
                        confidence=0.9,
                    )
                ],
                class_names={0: "person"},
            )
        }


class StubAlertsNode(Node):
    InputPorts = {"detections": "Detections"}
    OutputPorts = {"alerts": "Alerts"}

    def __init__(self, node_id="logic", config=None):
        super().__init__(node_id, config or {})
        self._state = NodeState.READY

    def process(self, inputs, context):
        return {
            "alerts": Alerts(items=[Alert(message=f"count={len(inputs['detections'].items)}")])
        }


class SparseNode(Node):
    InputPorts = {"frame_info": "FrameInfo"}
    OutputPorts = {"alerts": "Alerts"}

    def __init__(self, node_id="sparse", config=None):
        super().__init__(node_id, config or {})
        self._state = NodeState.READY

    def process(self, inputs, context):
        if inputs["frame_info"].frame_idx % 2 == 0:
            return {"alerts": Alerts(items=[Alert(message="tick")])}
        return {}


@pytest.fixture(autouse=True)
def register_stubs():
    saved = dict(NodeRegistry._registry)
    saved_categories = dict(NodeRegistry._categories)
    NodeRegistry._registry["stub_detector"] = StubDetectorNode
    NodeRegistry._registry["stub_alerts"] = StubAlertsNode
    NodeRegistry._registry["sparse"] = SparseNode
    NodeRegistry._categories["stub_detector"] = "testing"
    NodeRegistry._categories["stub_alerts"] = "testing"
    NodeRegistry._categories["sparse"] = "testing"
    yield
    NodeRegistry._registry.clear()
    NodeRegistry._registry.update(saved)
    NodeRegistry._categories.clear()
    NodeRegistry._categories.update(saved_categories)


def test_dag_dependencies_from_input_bindings():
    config = PipelineConfig(
        nodes=[
            NodeConfig(node_id="det", node_type="stub_detector", inputs={"image": "input.image"}),
            NodeConfig(
                node_id="logic",
                node_type="stub_alerts",
                inputs={"detections": "det.detections"},
            ),
        ]
    )
    dag = DAG(config)
    assert dag.dependencies("det") == []
    assert dag.dependencies("logic") == ["det"]
    assert dag.execution_order() == ["det", "logic"]


def test_dag_cycle_detection():
    config = PipelineConfig(
        nodes=[
            NodeConfig(node_id="a", node_type="stub_detector", inputs={"image": "b.image"}),
            NodeConfig(node_id="b", node_type="stub_detector", inputs={"image": "a.image"}),
        ]
    )
    dag = DAG(config)
    with pytest.raises(CycleError):
        dag.topological_sort()


def test_schema_validator_rejects_bad_type_edge():
    config = PipelineConfig(
        nodes=[
            NodeConfig(node_id="det", node_type="stub_detector", inputs={"image": "input.image"}),
            NodeConfig(node_id="logic", node_type="stub_alerts", inputs={"detections": "input.image"}),
        ]
    )
    dag = DAG(config)
    nodes = {"det": StubDetectorNode("det"), "logic": StubAlertsNode("logic")}
    with pytest.raises(SchemaValidationError, match="expects Detections"):
        SchemaValidator.validate(dag, nodes)


def test_executor_uses_selected_outputs():
    config = PipelineConfig(
        nodes=[
            NodeConfig(node_id="det", node_type="stub_detector", inputs={"image": "input.image"}),
            NodeConfig(
                node_id="logic",
                node_type="stub_alerts",
                inputs={"detections": "det.detections"},
            ),
        ],
        outputs={"alerts": "logic.alerts"},
    )
    dag = DAG(config)
    nodes = {"det": StubDetectorNode("det"), "logic": StubAlertsNode("logic")}
    executor = PipelineExecutor(dag, nodes)
    result = executor.run(np.zeros((8, 8, 3), dtype=np.uint8))
    assert list(result.keys()) == ["alerts"]
    assert result["alerts"].items[0].message == "count=1"


def test_executor_skips_missing_outputs():
    config = PipelineConfig(
        nodes=[
            NodeConfig(node_id="sparse", node_type="sparse", inputs={"frame_info": "input.frame_info"}),
        ],
        outputs={"alerts": "sparse.alerts"},
    )
    dag = DAG(config)
    nodes = {"sparse": SparseNode("sparse")}
    executor = PipelineExecutor(dag, nodes)
    even_result = executor.run(np.zeros((4, 4, 3), dtype=np.uint8), NodeContext(frame_idx=0))
    odd_result = executor.run(np.zeros((4, 4, 3), dtype=np.uint8), NodeContext(frame_idx=1))
    assert "alerts" in even_result
    assert odd_result == {}


def test_pipeline_end_to_end():
    config = PipelineConfig(
        nodes=[
            NodeConfig(node_id="det", node_type="stub_detector", inputs={"image": "input.image"}),
            NodeConfig(
                node_id="logic",
                node_type="stub_alerts",
                inputs={"detections": "det.detections"},
            ),
        ],
        outputs={"alerts": "logic.alerts"},
    )
    pipeline = Pipeline(config)
    result = pipeline.run(np.zeros((16, 16, 3), dtype=np.uint8))
    assert result["alerts"].items[0].message == "count=1"
