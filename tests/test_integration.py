"""
Stage 6: Integration Tests
============================

Tests the end-to-end flow: config → Pipeline → run → results.
Uses stub nodes — no real ML models.

Run: pytest tests/test_integration.py -v
"""
import pytest
import numpy as np

from vision_tools.core.config import NodeConfig, PipelineConfig, ExecutionConfig
from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.schemas import (
    DetectionResult, BoundingBox,
    EmbeddingResult, Embedding,
)
from vision_tools.pipeline import Pipeline
from vision_tools.nodes.logic_node import LogicNode


# ===================================================================
# Stub nodes
# ===================================================================

class IntegrationDetector(Node):
    """Stub detector for integration testing."""
    OutputSchema = DetectionResult
    InputSchema = None

    def __init__(self, node_id="det", config=None):
        super().__init__(node_id, config or {})
        self._state = NodeState.READY

    def process(self, data, context):
        return DetectionResult(
            boxes=[
                BoundingBox(xyxy=[0, 0, 100, 100], class_id=0,
                           confidence=0.95, class_name="person"),
                BoundingBox(xyxy=[50, 50, 200, 200], class_id=1,
                           confidence=0.3, class_name="car"),
            ],
            class_names={0: "person", 1: "car"},
        ).model_dump()

    def load(self): self._state = NodeState.READY
    def unload(self): self._state = NodeState.UNLOADED


class IntegrationEmbedder(Node):
    """Stub embedder for integration testing."""
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


class ConfidenceFilter(LogicNode):
    """Filter detections below threshold."""
    InputSchema = DetectionResult
    OutputSchema = DetectionResult

    def execute(self, data, context):
        threshold = self.config.get("threshold", 0.5)
        boxes = [b for b in data.get("boxes", []) if b["confidence"] >= threshold]
        return {"boxes": boxes, "class_names": data.get("class_names", {})}


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(autouse=True)
def register_integration_nodes():
    """Register stub nodes for integration testing."""
    saved = dict(NodeRegistry._registry)
    saved_cats = dict(NodeRegistry._categories)

    NodeRegistry._registry["integration_detector"] = IntegrationDetector
    NodeRegistry._registry["integration_embedder"] = IntegrationEmbedder
    NodeRegistry._registry["confidence_filter"] = ConfidenceFilter
    NodeRegistry._categories["integration_detector"] = "detection"
    NodeRegistry._categories["integration_embedder"] = "embedding"
    NodeRegistry._categories["confidence_filter"] = "logic"

    yield

    NodeRegistry._registry.clear()
    NodeRegistry._registry.update(saved)
    NodeRegistry._categories.clear()
    NodeRegistry._categories.update(saved_cats)


# ===================================================================
# Integration Tests
# ===================================================================

class TestEndToEndPipeline:
    """Test complete pipeline workflows."""

    def test_simple_detection_pipeline(self):
        """Single detector, load, run, shutdown."""
        config = PipelineConfig(
            name="simple_det",
            nodes=[
                NodeConfig(node_id="det", node_type="integration_detector",
                           config={}),
            ],
        )
        pipeline = Pipeline(config)
        pipeline.load()

        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = pipeline.run(frame)

        assert "det" in results
        assert len(results["det"]["boxes"]) == 2
        pipeline.shutdown()

    def test_detection_then_filter_pipeline(self):
        """Detector → ConfidenceFilter pipeline."""
        config = PipelineConfig(
            name="det_filter",
            nodes=[
                NodeConfig(node_id="det", node_type="integration_detector",
                           config={}),
                NodeConfig(node_id="filter", node_type="confidence_filter",
                           config={"threshold": 0.5},
                           depends_on=["det"]),
            ],
        )
        pipeline = Pipeline(config)
        pipeline.load()

        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = pipeline.run(frame)

        assert "det" in results
        assert "filter" in results
        # Det has 2 boxes (0.95 + 0.3), filter keeps only 0.95
        assert len(results["det"]["boxes"]) == 2
        assert len(results["filter"]["boxes"]) == 1
        assert results["filter"]["boxes"][0]["class_name"] == "person"
        pipeline.shutdown()

    def test_parallel_branches_pipeline(self):
        """det and emb run in parallel (no dependencies)."""
        config = PipelineConfig(
            name="parallel",
            nodes=[
                NodeConfig(node_id="det", node_type="integration_detector",
                           config={}),
                NodeConfig(node_id="emb", node_type="integration_embedder",
                           config={}),
            ],
            execution=ExecutionConfig(max_workers=2),
        )
        pipeline = Pipeline(config)
        pipeline.load()

        results = pipeline.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert "det" in results
        assert "emb" in results
        assert "boxes" in results["det"]
        assert "embedding" in results["emb"]
        pipeline.shutdown()

    def test_diamond_pipeline(self):
        """root → [det, emb] → filter."""
        config = PipelineConfig(
            name="diamond",
            nodes=[
                NodeConfig(node_id="det", node_type="integration_detector",
                           config={}),
                NodeConfig(node_id="emb", node_type="integration_embedder",
                           config={}),
                NodeConfig(node_id="filter", node_type="confidence_filter",
                           config={"threshold": 0.5},
                           depends_on=["det"]),
            ],
        )
        pipeline = Pipeline(config)
        pipeline.load()

        results = pipeline.run(np.zeros((10, 10, 3), dtype=np.uint8))
        assert len(results) == 3
        assert len(results["filter"]["boxes"]) == 1
        pipeline.shutdown()

    def test_batch_processing(self):
        """Process multiple frames."""
        config = PipelineConfig(
            name="batch",
            nodes=[
                NodeConfig(node_id="det", node_type="integration_detector",
                           config={}),
            ],
        )
        pipeline = Pipeline(config)
        pipeline.load()

        frames = [np.zeros((10, 10, 3), dtype=np.uint8)] * 5
        results = pipeline.run_batch(frames)
        assert len(results) == 5
        for r in results:
            assert "det" in r
        pipeline.shutdown()


class TestTopLevelImports:
    """Test that the public API is importable."""

    def test_core_imports(self):
        import vision_tools
        assert hasattr(vision_tools, "Node")
        assert hasattr(vision_tools, "NodeState")
        assert hasattr(vision_tools, "NodeContext")
        assert hasattr(vision_tools, "NodeRegistry")
        assert hasattr(vision_tools, "NodeConfig")
        assert hasattr(vision_tools, "PipelineConfig")

    def test_pipeline_imports(self):
        import vision_tools
        assert hasattr(vision_tools, "Pipeline")
        assert hasattr(vision_tools, "DAG")

    def test_node_imports(self):
        import vision_tools
        assert hasattr(vision_tools, "ModelNode")
        assert hasattr(vision_tools, "LogicNode")
        assert hasattr(vision_tools, "RemoteNode")

    def test_backend_imports(self):
        import vision_tools
        assert hasattr(vision_tools, "Backend")
        assert hasattr(vision_tools, "BackendRegistry")

    def test_schema_imports(self):
        import vision_tools
        assert hasattr(vision_tools, "DetectionResult")
        assert hasattr(vision_tools, "EmbeddingResult")
        assert hasattr(vision_tools, "CaptionResult")
        assert hasattr(vision_tools, "PoseResult")
        assert hasattr(vision_tools, "BatchPayload")

    def test_version(self):
        import vision_tools
        assert vision_tools.__version__ == "2.0.0"
