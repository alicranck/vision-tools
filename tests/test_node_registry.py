"""
Stage 3: Task Nodes + Model Backends Tests
============================================

Tests for task node registration/discovery and backend registry integration.
Uses mock backends to avoid loading real ML models.

Run: pytest tests/test_node_registry.py -v
"""
import pytest
from unittest.mock import patch, MagicMock

import numpy as np
from pydantic import BaseModel

from vision_tools.backends.base import Backend
from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.schemas import (
    DetectionResult, BoundingBox,
    EmbeddingResult, Embedding,
    CaptionResult, Caption,
    PoseResult, PoseKeypoints, Keypoint,
)


# ===================================================================
# Mock backends that produce valid schema output
# ===================================================================

class MockDetectionBackend:
    def __init__(self):
        self.config = {}

    def configure(self, config):
        self.config = dict(config)

    def load_model(self, model_path, device="auto"):
        return {"mock": True}
    def infer(self, model, inputs):
        return {"raw": True}
    def postprocess(self, raw_output, context):
        return DetectionResult(
            boxes=[BoundingBox(xyxy=[0, 0, 100, 100], class_id=0,
                              confidence=0.9, class_name="person")],
            class_names={0: "person"}
        ).model_dump()
    def get_available_runtimes(self):
        return ["mock"]


class MockEmbeddingBackend:
    def load_model(self, model_path, device="auto"):
        return {"mock": True}
    def infer(self, model, inputs):
        return np.random.randn(1, 768)
    def postprocess(self, raw_output, context):
        vector = [0.1] * 768
        emb = Embedding(vector=vector, model_id="mock", dimension=768)
        return EmbeddingResult(embedding=emb).model_dump()
    def encode_text(self, model, text):
        return [0.1] * 768
    def get_available_runtimes(self):
        return ["mock"]


class MockCaptioningBackend:
    def load_model(self, model_path, device="auto"):
        return {"mock": True}
    def infer(self, model, inputs):
        return ["A mock caption"]
    def postprocess(self, raw_output, context):
        return CaptionResult(
            caption=Caption(text="A mock caption", model_id="mock")
        ).model_dump()
    def get_available_runtimes(self):
        return ["mock"]


class MockPoseBackend:
    def load_model(self, model_path, device="auto"):
        return {"mock": True}
    def infer(self, model, inputs):
        return {"raw": True}
    def postprocess(self, raw_output, context):
        kpts = [Keypoint(x=0.0, y=0.0, confidence=1.0)] * 17
        return PoseResult(
            poses=[PoseKeypoints(person_id=0, keypoints=kpts)]
        ).model_dump()
    def get_available_runtimes(self):
        return ["mock"]


# ===================================================================
# Fixtures: register mock backends before each test
# ===================================================================

@pytest.fixture(autouse=True)
def clean_registries():
    """Clear registries before/after each test."""
    # Save and clear
    saved_backends = dict(BackendRegistry._registry)
    saved_nodes = dict(NodeRegistry._registry)
    saved_categories = dict(NodeRegistry._categories)
    BackendRegistry.clear()
    NodeRegistry._registry.clear()
    NodeRegistry._categories.clear()

    # Register mock backends
    BackendRegistry._registry[("detection", "yolo")] = MockDetectionBackend
    BackendRegistry._registry[("embedding", "siglip2")] = MockEmbeddingBackend
    BackendRegistry._registry[("embedding", "clip")] = MockEmbeddingBackend
    BackendRegistry._registry[("captioning", "smolvlm")] = MockCaptioningBackend
    BackendRegistry._registry[("captioning", "llamacpp")] = MockCaptioningBackend
    BackendRegistry._registry[("pose", "yolo_pose")] = MockPoseBackend

    # Import nodes to trigger registration (they use @NodeRegistry.register)
    from vision_tools.nodes.detection import ObjectDetector
    from vision_tools.nodes.embedding import Embedder
    from vision_tools.nodes.captioning import Captioner
    from vision_tools.nodes.pose import PoseEstimator

    # Re-register if cleared
    if "object_detector" not in NodeRegistry._registry:
        NodeRegistry._registry["object_detector"] = ObjectDetector
        NodeRegistry._categories["object_detector"] = "detection"
    if "embedder" not in NodeRegistry._registry:
        NodeRegistry._registry["embedder"] = Embedder
        NodeRegistry._categories["embedder"] = "embedding"
    if "captioner" not in NodeRegistry._registry:
        NodeRegistry._registry["captioner"] = Captioner
        NodeRegistry._categories["captioner"] = "captioning"
    if "pose_estimator" not in NodeRegistry._registry:
        NodeRegistry._registry["pose_estimator"] = PoseEstimator
        NodeRegistry._categories["pose_estimator"] = "pose"

    yield

    # Restore both _registry and _categories
    BackendRegistry._registry.clear()
    BackendRegistry._registry.update(saved_backends)
    NodeRegistry._registry.clear()
    NodeRegistry._registry.update(saved_nodes)
    NodeRegistry._categories.clear()
    NodeRegistry._categories.update(saved_categories)


# ===================================================================
# 1. Node Discovery / Registration Tests
# ===================================================================

class TestNodeDiscovery:
    """Test that all task nodes are discoverable through the registry."""

    def test_all_nodes_registered(self):
        nodes = NodeRegistry.list_nodes()
        names = {n["type"] for n in nodes}
        assert "object_detector" in names
        assert "embedder" in names
        assert "captioner" in names
        assert "pose_estimator" in names

    def test_node_categories(self):
        nodes = NodeRegistry.list_nodes()
        cats = {n["type"]: n["category"] for n in nodes}
        assert cats["object_detector"] == "detection"
        assert cats["embedder"] == "embedding"
        assert cats["captioner"] == "captioning"
        assert cats["pose_estimator"] == "pose"

    def test_node_output_schemas(self):
        from vision_tools.nodes.detection import ObjectDetector
        from vision_tools.nodes.embedding import Embedder
        from vision_tools.nodes.captioning import Captioner
        from vision_tools.nodes.pose import PoseEstimator

        assert ObjectDetector.OutputSchema is DetectionResult
        assert Embedder.OutputSchema is EmbeddingResult
        assert Captioner.OutputSchema is CaptionResult
        assert PoseEstimator.OutputSchema is PoseResult

    def test_node_config_schemas(self):
        from vision_tools.nodes.detection import ObjectDetector, ObjectDetectorConfig
        from vision_tools.nodes.embedding import Embedder, EmbedderConfig
        from vision_tools.nodes.captioning import Captioner, CaptionerConfig
        from vision_tools.nodes.pose import PoseEstimator, PoseEstimatorConfig

        assert ObjectDetector.get_config_schema() is ObjectDetectorConfig
        assert Embedder.get_config_schema() is EmbedderConfig
        assert Captioner.get_config_schema() is CaptionerConfig
        assert PoseEstimator.get_config_schema() is PoseEstimatorConfig


# ===================================================================
# 2. Node Creation via Registry
# ===================================================================

class TestNodeCreation:
    """Test creating nodes through the registry factory."""

    def test_create_detector(self):
        from vision_tools.core.config import NodeConfig
        config = NodeConfig(
            node_id="det1",
            node_type="object_detector",
            config={
                "task": "detection",
                "model_family": "yolo",
                "size": "small",
                "device": "cpu",
                "vocabulary": ["person"],
            },
        )
        node = NodeRegistry.create(config)
        assert node.node_id == "det1"
        assert node.state == NodeState.UNLOADED

    def test_create_embedder(self):
        from vision_tools.core.config import NodeConfig
        config = NodeConfig(
            node_id="emb1",
            node_type="embedder",
            config={"model": "siglip2"}
        )
        node = NodeRegistry.create(config)
        assert node.node_id == "emb1"

    def test_create_captioner(self):
        from vision_tools.core.config import NodeConfig
        config = NodeConfig(
            node_id="cap1",
            node_type="captioner",
            config={"model": "smolvlm"}
        )
        node = NodeRegistry.create(config)
        assert node.node_id == "cap1"

    def test_create_pose(self):
        from vision_tools.core.config import NodeConfig
        config = NodeConfig(
            node_id="pose1",
            node_type="pose_estimator",
            config={"model": "yolo_pose"}
        )
        node = NodeRegistry.create(config)
        assert node.node_id == "pose1"


# ===================================================================
# 3. Task Node Lifecycle (with mock backends)
# ===================================================================

class TestTaskNodeLifecycle:
    """Test load → process → unload lifecycle with mock backends."""

    def test_detector_lifecycle(self):
        from vision_tools.nodes.detection import ObjectDetector
        node = ObjectDetector(
            "det",
            {
                "task": "detection",
                "model_family": "yolo",
                "size": "small",
                "device": "cpu",
                "vocabulary": ["person"],
            },
        )
        assert node.state == NodeState.UNLOADED

        node.load()
        assert node.state == NodeState.READY

        result = node.process(np.zeros((640, 640, 3), dtype=np.uint8), NodeContext())
        assert "boxes" in result
        assert result["boxes"][0]["class_name"] == "person"

        node.unload()
        assert node.state == NodeState.UNLOADED

    def test_embedder_lifecycle(self):
        from vision_tools.nodes.embedding import Embedder
        node = Embedder("emb", {"model": "siglip2"})
        node.load()
        result = node.process(np.zeros((384, 384, 3), dtype=np.uint8), NodeContext())
        assert "embedding" in result
        assert len(result["embedding"]["vector"]) == 768
        node.unload()

    def test_captioner_lifecycle(self):
        from vision_tools.nodes.captioning import Captioner
        node = Captioner("cap", {"model": "smolvlm"})
        node.load()
        result = node.process(np.zeros((512, 512, 3), dtype=np.uint8), NodeContext())
        assert "caption" in result
        assert result["caption"]["text"] == "A mock caption"
        node.unload()

    def test_pose_lifecycle(self):
        from vision_tools.nodes.pose import PoseEstimator
        node = PoseEstimator("pose", {"model": "yolo_pose"})
        node.load()
        result = node.process(np.zeros((640, 640, 3), dtype=np.uint8), NodeContext())
        assert "poses" in result
        assert len(result["poses"]) == 1
        assert len(result["poses"][0]["keypoints"]) == 17
        node.unload()


# ===================================================================
# 4. Embedder.encode_text
# ===================================================================

class TestEmbedderEncodeText:
    """Test text encoding via backend passthrough."""

    def test_encode_text(self):
        from vision_tools.nodes.embedding import Embedder
        node = Embedder("emb", {"model": "siglip2"})
        node.load()
        result = node.encode_text("hello world")
        assert isinstance(result, list)
        assert len(result) == 768
        node.unload()


# ===================================================================
# 5. Backend Registry Integration
# ===================================================================

class TestBackendRegistryIntegration:
    """Test that backends are correctly resolved from task node config."""

    def test_detector_resolves_yolo(self):
        from vision_tools.nodes.detection import ObjectDetector
        node = ObjectDetector(
            "det",
            {
                "task": "detection",
                "model_family": "yolo",
                "size": "small",
                "device": "cpu",
            },
        )
        assert isinstance(node.backend, MockDetectionBackend)
        assert node.backend.config.get("model_family") == "yolo"
        assert node.backend.config.get("checkpoint_id")

    def test_embedder_resolves_siglip2(self):
        from vision_tools.nodes.embedding import Embedder
        node = Embedder("emb", {"model": "siglip2"})
        assert isinstance(node.backend, MockEmbeddingBackend)

    def test_embedder_resolves_clip(self):
        from vision_tools.nodes.embedding import Embedder
        node = Embedder("emb", {"model": "clip"})
        assert isinstance(node.backend, MockEmbeddingBackend)

    def test_unknown_backend_raises(self):
        from vision_tools.nodes.detection import ObjectDetector
        with pytest.raises(ValueError):
            ObjectDetector(
                "det",
                {
                    "task": "detection",
                    "model_family": "nonexistent_model",
                    "size": "small",
                    "device": "cpu",
                },
            )


# ===================================================================
# 6. Metadata for LLM Agent Introspection
# ===================================================================

class TestLLMIntrospection:
    """Test that nodes expose metadata usable by LLM agents."""

    def test_node_metadata(self):
        from vision_tools.nodes.detection import ObjectDetector
        meta = ObjectDetector.get_metadata()
        assert "description" in meta
        assert "output_schema" in meta
        assert "config_schema" in meta
        assert "config_options" in meta
        assert "model_catalog" in meta["config_options"]

    def test_config_schema_as_json(self):
        from vision_tools.nodes.detection import ObjectDetector
        schema = ObjectDetector.get_config_schema().model_json_schema()
        assert "task" in schema["properties"]
        assert "model_family" in schema["properties"]
        assert "size" in schema["properties"]
        assert "device" in schema["properties"]
        assert "vocabulary" in schema["properties"]
        assert "conf_threshold" in schema["properties"]
