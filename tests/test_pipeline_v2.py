"""
Stage 4: Pipeline Split Tests
===============================

Tests for DAG, SchemaValidator, PipelineExecutor, and Pipeline facade.
Uses mock nodes — no real ML models.

Run: pytest tests/test_pipeline_v2.py -v
"""
import pytest
import numpy as np
from pydantic import BaseModel

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.config import NodeConfig, PipelineConfig, ExecutionConfig
from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.schemas import DetectionResult, BoundingBox, EmbeddingResult, Embedding
from vision_tools.pipeline import Pipeline
from vision_tools.pipeline.graph import DAG, CycleError
from vision_tools.pipeline.validator import SchemaValidator, SchemaValidationError
from vision_tools.pipeline.executor import PipelineExecutor


# ===================================================================
# Stub Nodes for Pipeline Testing
# ===================================================================

class StubDetectorNode(Node):
    """Stub detector that returns valid DetectionResult."""
    OutputSchema = DetectionResult
    InputSchema = None

    def __init__(self, node_id="det", config=None):
        super().__init__(node_id, config or {})
        self._state = NodeState.READY

    def process(self, data, context):
        return DetectionResult(
            boxes=[BoundingBox(xyxy=[0, 0, 100, 100], class_id=0,
                               confidence=0.9, class_name="person")],
            class_names={0: "person"}
        ).model_dump()

    def load(self): self._state = NodeState.READY
    def unload(self): self._state = NodeState.UNLOADED


class StubEmbedderNode(Node):
    """Stub embedder that returns valid EmbeddingResult."""
    OutputSchema = EmbeddingResult
    InputSchema = None

    def __init__(self, node_id="emb", config=None):
        super().__init__(node_id, config or {})
        self._state = NodeState.READY

    def process(self, data, context):
        vector = [0.1] * 768
        emb = Embedding(vector=vector, model_id="stub", dimension=768)
        return EmbeddingResult(embedding=emb).model_dump()

    def load(self): self._state = NodeState.READY
    def unload(self): self._state = NodeState.UNLOADED


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def simple_config():
    """Config with two independent nodes."""
    return PipelineConfig(
        name="test",
        nodes=[
            NodeConfig(node_id="det", node_type="stub_det", config={}),
            NodeConfig(node_id="emb", node_type="stub_emb", config={}),
        ],
    )

@pytest.fixture
def chain_config():
    """Config with a linear dependency chain: det → emb."""
    return PipelineConfig(
        name="chain_test",
        nodes=[
            NodeConfig(node_id="det", node_type="stub_det", config={}),
            NodeConfig(node_id="emb", node_type="stub_emb", config={},
                       depends_on=["det"]),
        ],
    )

@pytest.fixture
def diamond_config():
    """Diamond: root → A, B → sink."""
    return PipelineConfig(
        name="diamond",
        nodes=[
            NodeConfig(node_id="root", node_type="stub_det", config={}),
            NodeConfig(node_id="a", node_type="stub_det", config={},
                       depends_on=["root"]),
            NodeConfig(node_id="b", node_type="stub_emb", config={},
                       depends_on=["root"]),
            NodeConfig(node_id="sink", node_type="stub_emb", config={},
                       depends_on=["a", "b"]),
        ],
    )

@pytest.fixture(autouse=True)
def register_stubs():
    """Register stub nodes."""
    saved = dict(NodeRegistry._registry)
    saved_cats = dict(NodeRegistry._categories)
    NodeRegistry._registry["stub_det"] = StubDetectorNode
    NodeRegistry._registry["stub_emb"] = StubEmbedderNode
    NodeRegistry._categories["stub_det"] = "detection"
    NodeRegistry._categories["stub_emb"] = "embedding"
    yield
    NodeRegistry._registry.clear()
    NodeRegistry._registry.update(saved)
    NodeRegistry._categories.clear()
    NodeRegistry._categories.update(saved_cats)


# ===================================================================
# 1. DAG Tests
# ===================================================================

class TestDAG:
    """Test DAG construction, topological sort, and cycle detection."""

    def test_simple_dag_layers(self, simple_config):
        dag = DAG(simple_config)
        layers = dag.topological_sort()
        assert len(layers) == 1
        assert set(layers[0]) == {"det", "emb"}

    def test_chain_dag_layers(self, chain_config):
        dag = DAG(chain_config)
        layers = dag.topological_sort()
        assert len(layers) == 2
        assert layers[0] == ["det"]
        assert layers[1] == ["emb"]

    def test_diamond_dag_layers(self, diamond_config):
        dag = DAG(diamond_config)
        layers = dag.topological_sort()
        assert len(layers) == 3
        assert layers[0] == ["root"]
        assert set(layers[1]) == {"a", "b"}
        assert layers[2] == ["sink"]

    def test_execution_order(self, chain_config):
        dag = DAG(chain_config)
        order = dag.execution_order()
        assert order.index("det") < order.index("emb")

    def test_cycle_detection(self):
        config = PipelineConfig(
            name="cycle",
            nodes=[
                NodeConfig(node_id="a", node_type="stub_det",
                           config={}, depends_on=["b"]),
                NodeConfig(node_id="b", node_type="stub_det",
                           config={}, depends_on=["a"]),
            ],
        )
        dag = DAG(config)
        with pytest.raises(CycleError):
            dag.topological_sort()

    def test_missing_dependency(self):
        config = PipelineConfig(
            name="bad",
            nodes=[
                NodeConfig(node_id="a", node_type="stub_det",
                           config={}, depends_on=["nonexistent"]),
            ],
        )
        with pytest.raises(ValueError, match="not in the pipeline"):
            DAG(config)

    def test_dependencies(self, chain_config):
        dag = DAG(chain_config)
        assert dag.dependencies("emb") == ["det"]
        assert dag.dependencies("det") == []

    def test_dependents(self, chain_config):
        dag = DAG(chain_config)
        assert dag.dependents("det") == ["emb"]
        assert dag.dependents("emb") == []


# ===================================================================
# 2. SchemaValidator Tests
# ===================================================================

class TestSchemaValidator:
    """Test I/O schema validation."""

    def test_no_input_schema_passes(self, simple_config):
        dag = DAG(simple_config)
        nodes = {
            "det": StubDetectorNode("det"),
            "emb": StubEmbedderNode("emb"),
        }
        warnings = SchemaValidator.validate(dag, nodes)
        assert len(warnings) == 0

    def test_missing_output_schema_raises(self, chain_config):
        dag = DAG(chain_config)

        class NoOutputNode(Node):
            OutputSchema = None
            InputSchema = DetectionResult
            def __init__(self): super().__init__("no_out", {})
            def process(self, data, ctx): return {}

        class NeedsInputNode(Node):
            OutputSchema = None
            InputSchema = DetectionResult
            def __init__(self): super().__init__("emb", {})
            def process(self, data, ctx): return {}

        nodes = {"det": NoOutputNode(), "emb": NeedsInputNode()}
        with pytest.raises(SchemaValidationError, match="Schema validation failed"):
            SchemaValidator.validate(dag, nodes)

    def test_field_type_mismatch_raises(self):
        class ProducerOut(BaseModel):
            value: int

        class ConsumerIn(BaseModel):
            value: str

        class ProducerNode(Node):
            OutputSchema = ProducerOut
            InputSchema = None
            def __init__(self): super().__init__("prod", {})
            def process(self, data, ctx): return {"value": 1}

        class ConsumerNode(Node):
            OutputSchema = None
            InputSchema = ConsumerIn
            def __init__(self): super().__init__("cons", {})
            def process(self, data, ctx): return {}

        config = PipelineConfig(
            name="type_mismatch",
            nodes=[
                NodeConfig(node_id="prod", node_type="stub_det", config={}),
                NodeConfig(node_id="cons", node_type="stub_emb", config={}, depends_on=["prod"]),
            ],
        )
        dag = DAG(config)
        nodes = {"prod": ProducerNode(), "cons": ConsumerNode()}

        with pytest.raises(SchemaValidationError, match="type mismatch"):
            SchemaValidator.validate(dag, nodes)


# ===================================================================
# 3. PipelineExecutor Tests
# ===================================================================

class TestPipelineExecutor:
    """Test executor layer-by-layer processing."""

    def test_sequential_execution(self, chain_config):
        dag = DAG(chain_config)
        nodes = {"det": StubDetectorNode("det"), "emb": StubEmbedderNode("emb")}
        executor = PipelineExecutor(dag, nodes)

        results = executor.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert "det" in results
        assert "emb" in results
        assert "boxes" in results["det"]
        assert "embedding" in results["emb"]

    def test_parallel_execution(self, simple_config):
        dag = DAG(simple_config)
        nodes = {"det": StubDetectorNode("det"), "emb": StubEmbedderNode("emb")}
        executor = PipelineExecutor(dag, nodes, max_workers=2)

        results = executor.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert "det" in results
        assert "emb" in results

    def test_batch_execution(self, chain_config):
        dag = DAG(chain_config)
        nodes = {"det": StubDetectorNode("det"), "emb": StubEmbedderNode("emb")}
        executor = PipelineExecutor(dag, nodes)

        frames = [np.zeros((10, 10, 3), dtype=np.uint8)] * 3
        results = executor.run_batch(frames)
        assert len(results) == 3
        for r in results:
            assert "det" in r
            assert "emb" in r

    def test_upstream_results_injected(self, chain_config):
        """Verify that downstream nodes get upstream results in context."""
        captured = {}

        class CapturingNode(Node):
            OutputSchema = EmbeddingResult
            InputSchema = None
            def __init__(self):
                super().__init__("emb", {})
                self._state = NodeState.READY
            def process(self, data, context):
                captured["upstream"] = context.upstream_results
                vector = [0.1] * 768
                emb = Embedding(vector=vector, model_id="stub", dimension=768)
                return EmbeddingResult(embedding=emb).model_dump()

        dag = DAG(chain_config)
        nodes = {"det": StubDetectorNode("det"), "emb": CapturingNode()}
        executor = PipelineExecutor(dag, nodes)

        executor.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert "det" in captured["upstream"]
        assert "boxes" in captured["upstream"]["det"]


# ===================================================================
# 4. Pipeline Facade Tests
# ===================================================================

class TestPipelineFacade:
    """Test the Pipeline high-level API."""

    def test_create_pipeline(self, chain_config):
        pipeline = Pipeline(chain_config)
        assert "det" in pipeline.nodes
        assert "emb" in pipeline.nodes

    def test_load_and_run(self, chain_config):
        pipeline = Pipeline(chain_config)
        pipeline.load()
        results = pipeline.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert "det" in results
        assert "emb" in results

    def test_run_batch(self, chain_config):
        pipeline = Pipeline(chain_config)
        pipeline.load()
        results = pipeline.run_batch(
            [np.zeros((10, 10, 3), dtype=np.uint8)] * 2
        )
        assert len(results) == 2

    def test_shutdown(self, chain_config):
        pipeline = Pipeline(chain_config)
        pipeline.load()
        pipeline.shutdown()
        # Nodes should be unloaded
        for node in pipeline.nodes.values():
            assert node.state == NodeState.UNLOADED

    def test_get_node(self, chain_config):
        pipeline = Pipeline(chain_config)
        node = pipeline.get_node("det")
        assert node.node_id == "det"

    def test_get_node_missing(self, chain_config):
        pipeline = Pipeline(chain_config)
        with pytest.raises(KeyError, match="not found"):
            pipeline.get_node("nonexistent")

    def test_repr(self, chain_config):
        pipeline = Pipeline(chain_config)
        r = repr(pipeline)
        assert "chain_test" in r
        assert "det" in r

    def test_diamond_pipeline(self, diamond_config):
        pipeline = Pipeline(diamond_config)
        pipeline.load()
        results = pipeline.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert len(results) == 4
        assert "root" in results
        assert "a" in results
        assert "b" in results
        assert "sink" in results
        pipeline.shutdown()

    def test_execution_config(self):
        config = PipelineConfig(
            name="par",
            nodes=[
                NodeConfig(node_id="a", node_type="stub_det", config={}),
                NodeConfig(node_id="b", node_type="stub_emb", config={}),
            ],
            execution=ExecutionConfig(max_workers=2, warmup_rounds=0),
        )
        pipeline = Pipeline(config)
        pipeline.load()
        results = pipeline.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert "a" in results
        assert "b" in results
        pipeline.shutdown()
