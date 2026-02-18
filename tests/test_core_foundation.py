"""
Stage 1: Core Foundation Tests
===============================

Tests for the new v2 core: Node ABC, NodeRegistry, NodeConfig, and schemas.

Run: pytest tests/test_core_foundation.py -v
"""
import pytest
from pydantic import BaseModel, ValidationError

from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.config import NodeConfig, ExecutionConfig, PipelineConfig


# ===================================================================
# 1. Node ABC Tests
# ===================================================================

class TestNodeState:
    """Test NodeState enum."""

    def test_all_states_exist(self):
        assert NodeState.UNLOADED == "unloaded"
        assert NodeState.READY == "ready"
        assert NodeState.FAILED == "failed"
        assert NodeState.TRAINING == "training"

    def test_states_are_strings(self):
        for state in NodeState:
            assert isinstance(state.value, str)


class TestNodeContext:
    """Test NodeContext Pydantic model."""

    def test_defaults(self):
        ctx = NodeContext()
        assert ctx.frame_idx == 0
        assert ctx.timestamp == 0.0
        assert ctx.camera_id == "default"
        assert ctx.upstream_results == {}

    def test_custom_context(self):
        ctx = NodeContext(
            frame_idx=42,
            timestamp=1.5,
            frame_shape=(480, 640, 3),
            camera_id="cam-01",
            upstream_results={"detector": {"boxes": []}},
        )
        assert ctx.frame_idx == 42
        assert ctx.upstream_results["detector"] == {"boxes": []}

    def test_serialization(self):
        ctx = NodeContext(frame_idx=1, timestamp=0.5)
        d = ctx.model_dump()
        restored = NodeContext.model_validate(d)
        assert restored.frame_idx == 1


class DummyOutput(BaseModel):
    value: int


class ConcreteNode(Node):
    """A minimal concrete Node for testing."""
    OutputSchema = DummyOutput

    def process(self, data, context):
        return {"value": data * 2}


class TestNode:
    """Test the Node ABC."""

    def test_instantiation(self):
        node = ConcreteNode(node_id="test")
        assert node.node_id == "test"
        assert node.state == NodeState.READY
        assert node.config == {}

    def test_with_config(self):
        node = ConcreteNode(node_id="test", config={"key": "val"})
        assert node.config["key"] == "val"

    def test_process(self):
        node = ConcreteNode(node_id="test")
        result = node.process(21, NodeContext())
        assert result == {"value": 42}

    def test_repr(self):
        node = ConcreteNode(node_id="my_node")
        r = repr(node)
        assert "my_node" in r
        assert "ready" in r

    def test_schemas_none_by_default(self):
        assert ConcreteNode.InputSchema is None
        assert ConcreteNode.OutputSchema is DummyOutput

    def test_get_metadata(self):
        meta = ConcreteNode.get_metadata()
        assert "description" in meta
        assert meta["output_schema"] is not None
        assert meta["input_schema"] is None

    def test_get_config_schema_default_none(self):
        assert ConcreteNode.get_config_schema() is None

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Node(node_id="fail")


# ===================================================================
# 2. NodeRegistry Tests
# ===================================================================

class TestNodeRegistry:
    """Test NodeRegistry registration and discovery."""

    def setup_method(self):
        """Clear registry before each test."""
        NodeRegistry.clear()

    def test_register_decorator(self):
        @NodeRegistry.register("test_node", category="testing")
        class TestNode(ConcreteNode):
            pass

        assert "test_node" in NodeRegistry.registered_names()

    def test_get_registered_node(self):
        @NodeRegistry.register("my_node")
        class MyNode(ConcreteNode):
            pass

        assert NodeRegistry.get("my_node") is MyNode

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown node type"):
            NodeRegistry.get("nonexistent")

    def test_list_nodes(self):
        @NodeRegistry.register("node_a", category="cat_1")
        class NodeA(ConcreteNode):
            pass

        @NodeRegistry.register("node_b", category="cat_2")
        class NodeB(ConcreteNode):
            pass

        nodes = NodeRegistry.list_nodes()
        assert len(nodes) == 2
        types = {n["type"] for n in nodes}
        assert types == {"node_a", "node_b"}
        categories = {n["category"] for n in nodes}
        assert categories == {"cat_1", "cat_2"}

    def test_create_from_config(self):
        @NodeRegistry.register("creator_test")
        class CreatorTestNode(ConcreteNode):
            pass

        config = NodeConfig(
            node_id="my_instance",
            node_type="creator_test",
            config={"param": 42},
        )
        node = NodeRegistry.create(config)
        assert node.node_id == "my_instance"
        assert node.config["param"] == 42
        assert isinstance(node, CreatorTestNode)

    def test_create_unknown_type_raises(self):
        config = NodeConfig(node_id="x", node_type="nonexistent")
        with pytest.raises(KeyError):
            NodeRegistry.create(config)

    def test_clear(self):
        @NodeRegistry.register("temp")
        class TempNode(ConcreteNode):
            pass

        assert len(NodeRegistry.registered_names()) == 1
        NodeRegistry.clear()
        assert len(NodeRegistry.registered_names()) == 0

    def test_overwrite_warning(self, caplog):
        @NodeRegistry.register("dup")
        class First(ConcreteNode):
            pass

        import logging
        with caplog.at_level(logging.WARNING):
            @NodeRegistry.register("dup")
            class Second(ConcreteNode):
                pass

        assert NodeRegistry.get("dup") is Second


# ===================================================================
# 3. Config Tests
# ===================================================================

class TestNodeConfig:
    """Test NodeConfig validation."""

    def test_basic_config(self):
        nc = NodeConfig(node_id="det", node_type="object_detector")
        assert nc.node_id == "det"
        assert nc.node_type == "object_detector"
        assert nc.config == {}
        assert nc.depends_on == []

    def test_with_dependencies(self):
        nc = NodeConfig(
            node_id="captioner",
            node_type="captioner",
            config={"model": "smolvlm"},
            depends_on=["detector"],
        )
        assert nc.depends_on == ["detector"]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            NodeConfig(
                node_id="test",
                node_type="test",
                unknown_field="bad",
            )

    def test_serialization(self):
        nc = NodeConfig(node_id="a", node_type="b", config={"x": 1})
        d = nc.model_dump()
        restored = NodeConfig.model_validate(d)
        assert restored == nc


class TestExecutionConfig:
    """Test ExecutionConfig validation."""

    def test_defaults(self):
        ec = ExecutionConfig()
        assert ec.mode == "sequential"
        assert ec.max_workers == 4
        assert ec.warmup_rounds == 4
        assert ec.verify_on_init is True

    def test_parallel_mode(self):
        ec = ExecutionConfig(mode="parallel", max_workers=8)
        assert ec.mode == "parallel"

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(mode="distributed")

    def test_max_workers_bounds(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(max_workers=0)
        with pytest.raises(ValidationError):
            ExecutionConfig(max_workers=100)


class TestPipelineConfig:
    """Test PipelineConfig validation."""

    def test_basic_pipeline(self):
        pc = PipelineConfig(
            name="test_pipeline",
            nodes=[
                NodeConfig(node_id="det", node_type="object_detector"),
            ],
        )
        assert pc.name == "test_pipeline"
        assert len(pc.nodes) == 1

    def test_requires_at_least_one_node(self):
        with pytest.raises(ValidationError):
            PipelineConfig(name="empty", nodes=[])

    def test_multi_node_dag(self):
        pc = PipelineConfig(
            nodes=[
                NodeConfig(node_id="det", node_type="object_detector"),
                NodeConfig(
                    node_id="emb",
                    node_type="embedder",
                    depends_on=["det"],
                ),
            ],
            execution=ExecutionConfig(mode="parallel"),
        )
        assert len(pc.nodes) == 2
        assert pc.nodes[1].depends_on == ["det"]
        assert pc.execution.mode == "parallel"

    def test_serialization_roundtrip(self):
        pc = PipelineConfig(
            name="roundtrip",
            nodes=[
                NodeConfig(node_id="a", node_type="x"),
                NodeConfig(node_id="b", node_type="y", depends_on=["a"]),
            ],
        )
        d = pc.model_dump()
        restored = PipelineConfig.model_validate(d)
        assert restored == pc

    def test_json_roundtrip(self):
        pc = PipelineConfig(
            nodes=[NodeConfig(node_id="a", node_type="x")],
        )
        json_str = pc.model_dump_json()
        restored = PipelineConfig.model_validate_json(json_str)
        assert restored == pc


# ===================================================================
# 4. Schema re-export test
# ===================================================================

class TestSchemaReExport:
    """Verify backward-compat re-export works."""

    def test_import_from_old_path(self):
        from vision_tools.utils.schemas import DetectionResult, BoundingBox
        from vision_tools.core.schemas import DetectionResult as DR2

        assert DetectionResult is DR2

    def test_import_from_new_path(self):
        from vision_tools.core.schemas import (
            DetectionResult,
            BoundingBox,
            EmbeddingResult,
            PoseResult,
            CaptionResult,
            FrameMetadata,
            FrameResult,
            BatchPayload,
        )
        # Sanity check
        dr = DetectionResult()
        assert dr.boxes == []
