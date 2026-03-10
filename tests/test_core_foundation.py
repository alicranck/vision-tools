import pytest
from pydantic import BaseModel

from vision_tools.core.config import ExecutionConfig, NodeConfig, PipelineConfig
from vision_tools.core.graph_types import Detections
from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.type_refs import (
    GenericTypeRef,
    SimpleTypeRef,
    TypeRegistry,
    parse_type_ref,
)


class DummyOutput(BaseModel):
    value: int


class ConcreteNode(Node):
    InputPorts = {"detections": "Detections"}
    OutputPorts = {"detections": "Detections"}

    def process(self, inputs, context):
        return {"detections": inputs["detections"]}


def test_node_context_defaults():
    ctx = NodeContext()
    assert ctx.frame_idx == 0
    assert ctx.camera_id == "default"
    assert ctx.port_values == {}


def test_node_state_values():
    assert NodeState.UNLOADED == "unloaded"
    assert NodeState.READY == "ready"
    assert NodeState.FAILED == "failed"
    assert NodeState.TRAINING == "training"


def test_node_metadata_uses_ports():
    metadata = ConcreteNode.get_metadata()
    assert metadata["input_ports"]["detections"] == "Detections"
    assert metadata["output_ports"]["detections"] == "Detections"
    assert metadata["dynamic_ports"] is False


def test_type_ref_parsing():
    assert parse_type_ref("Image") == SimpleTypeRef("Image")
    assert parse_type_ref("History[Tracks]") == GenericTypeRef(
        "History", SimpleTypeRef("Tracks")
    )


def test_type_registry_assignable_union_target():
    assert TypeRegistry.is_assignable("Detections", "Detections | Tracks")
    assert not TypeRegistry.is_assignable("Tracks", "Detections")


def test_node_registry_create():
    NodeRegistry.clear()

    @NodeRegistry.register("concrete", category="testing")
    class RegisteredConcrete(ConcreteNode):
        pass

    node = NodeRegistry.create(
        NodeConfig(node_id="n1", node_type="concrete", inputs={"detections": "input.image"})
    )
    assert isinstance(node, RegisteredConcrete)
    assert node.node_id == "n1"


def test_pipeline_config_serialization():
    config = PipelineConfig(
        name="test",
        nodes=[
            NodeConfig(
                node_id="filter",
                node_type="filter",
                config={"min_confidence": 0.5},
                inputs={"detections": "det.detections"},
            )
        ],
        outputs={"detections": "filter.detections"},
    )
    restored = PipelineConfig.model_validate(config.model_dump())
    assert restored == config


def test_execution_config_defaults():
    cfg = ExecutionConfig()
    assert cfg.mode == "sequential"
    assert cfg.max_workers == 1
    assert cfg.verify_on_init is False
