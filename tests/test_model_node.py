"""
Stage 2: ModelNode + Backend + Runtime Tests
=============================================

Tests for ModelNode, Backend protocol, BackendRegistry, ModelResolver, ModelCache.
All tests use mock/stub backends — no real ML models loaded.

Run: pytest tests/test_model_node.py -v
"""
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from vision_tools.backends.base import Backend
from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext, NodeState
from vision_tools.core.schemas import DetectionResult, BoundingBox
from vision_tools.nodes.model_node import ModelNode
from vision_tools.runtime.model_cache import ModelCache
from vision_tools.runtime.model_resolver import ModelResolver


# ===================================================================
# Stub Backend for testing
# ===================================================================

class StubBackend:
    """Minimal backend that satisfies the Backend protocol."""

    def load_model(self, model_path, device="auto"):
        return {"loaded_from": model_path, "device": device}

    def infer(self, model, inputs):
        return {"raw": "output", "input_shape": getattr(inputs, "shape", None)}

    def postprocess(self, raw_output, context):
        return {
            "boxes": [
                {
                    "xyxy": [0.0, 0.0, 100.0, 100.0],
                    "class_id": 0,
                    "confidence": 0.95,
                    "class_name": "person",
                }
            ]
        }

    def get_available_runtimes(self):
        return ["pytorch", "openvino"]


class StrictOutput(BaseModel):
    """A strict schema for negative testing."""
    required_field: str  # This field is required, no default


class BadOutputBackend:
    """Backend that returns output incompatible with StrictOutput."""

    def load_model(self, model_path, device="auto"):
        return "model"

    def infer(self, model, inputs):
        return "raw"

    def postprocess(self, raw_output, context):
        return {"wrong_field": "bad_data"}

    def get_available_runtimes(self):
        return ["cpu"]


# ===================================================================
# Stub TaskNode for testing
# ===================================================================

class StubDetector(ModelNode):
    """A task node that uses StubBackend for testing."""
    OutputSchema = DetectionResult

    def __init__(self, node_id="test_det", config=None, **kwargs):
        config = config or {"model": "test_model"}
        kwargs.setdefault("backend", StubBackend())
        super().__init__(node_id=node_id, config=config, **kwargs)


class StubDetectorBadOutput(ModelNode):
    """Task node with a backend that produces invalid output."""
    OutputSchema = StrictOutput

    def __init__(self, node_id="bad_det", config=None, **kwargs):
        config = config or {"model": "test_model"}
        kwargs.setdefault("backend", BadOutputBackend())
        super().__init__(node_id=node_id, config=config, **kwargs)


# ===================================================================
# 1. Backend Protocol Tests
# ===================================================================

class TestBackendProtocol:
    """Verify the Backend protocol works with runtime_checkable."""

    def test_stub_satisfies_protocol(self):
        assert isinstance(StubBackend(), Backend)

    def test_dict_does_not_satisfy(self):
        assert not isinstance({}, Backend)


# ===================================================================
# 2. BackendRegistry Tests
# ===================================================================

class TestBackendRegistry:
    """Test BackendRegistry registration and lookup."""

    def setup_method(self):
        BackendRegistry.clear()

    def test_register_and_get(self):
        @BackendRegistry.register(task="detection", model="test")
        class TestBackend:
            def load_model(self, model_path, device="auto"): return None
            def infer(self, model, inputs): return None
            def postprocess(self, raw_output, context): return {}
            def get_available_runtimes(self): return []

        backend = BackendRegistry.get("detection", "test")
        assert isinstance(backend, TestBackend)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="No backend registered"):
            BackendRegistry.get("nonexistent", "nope")

    def test_list_backends(self):
        @BackendRegistry.register(task="detection", model="yolo")
        class YoloB:
            pass

        @BackendRegistry.register(task="embedding", model="siglip2")
        class SigB:
            pass

        backends = BackendRegistry.list_backends()
        assert len(backends) == 2
        tasks = {b["task"] for b in backends}
        assert tasks == {"detection", "embedding"}

    def test_has(self):
        @BackendRegistry.register(task="test", model="x")
        class X:
            pass

        assert BackendRegistry.has("test", "x")
        assert not BackendRegistry.has("test", "y")

    def test_clear(self):
        @BackendRegistry.register(task="a", model="b")
        class AB:
            pass

        assert BackendRegistry.has("a", "b")
        BackendRegistry.clear()
        assert not BackendRegistry.has("a", "b")


# ===================================================================
# 3. ModelNode Tests
# ===================================================================

class TestModelNode:
    """Test ModelNode lifecycle, processing, and schema enforcement."""

    def test_initial_state_unloaded(self):
        node = StubDetector()
        assert node.state == NodeState.UNLOADED

    def test_load(self):
        node = StubDetector()
        node.load()
        assert node.state == NodeState.READY
        assert node.model is not None

    def test_load_idempotent(self):
        node = StubDetector()
        node.load()
        node.load()  # should not error
        assert node.state == NodeState.READY

    def test_unload(self):
        node = StubDetector()
        node.load()
        node.unload()
        assert node.state == NodeState.UNLOADED
        assert node.model is None

    def test_process_requires_load(self):
        node = StubDetector()
        with pytest.raises(RuntimeError, match="cannot process"):
            node.process(np.zeros((10, 10, 3)), NodeContext())

    def test_process_returns_validated(self):
        node = StubDetector()
        node.load()
        result = node.process(np.zeros((10, 10, 3)), NodeContext())
        assert "boxes" in result
        assert len(result["boxes"]) == 1
        assert result["boxes"][0]["class_name"] == "person"

    def test_process_schema_enforcement(self):
        node = StubDetectorBadOutput()
        node.load()
        # BadOutputBackend returns {"invalid_field": "bad_data"}
        # which should fail DetectionResult validation
        with pytest.raises(ValidationError):
            node.process(np.zeros((10, 10, 3)), NodeContext())

    def test_no_backend_raises_on_load(self):
        node = ModelNode.__new__(ModelNode)
        node.node_id = "test"
        node.config = {"model": "x"}
        node._state = NodeState.UNLOADED
        node.model = None
        node.backend = None
        node.model_cache = ModelCache()
        node.model_resolver = ModelResolver(gpu_vram_gb=0.0)
        with pytest.raises(RuntimeError, match="no backend set"):
            node.load()

    def test_warmup(self):
        node = StubDetector()
        node.load()
        # Should not raise
        node.warmup(rounds=2)

    def test_warmup_not_ready(self):
        node = StubDetector()
        # Should log warning, not raise
        node.warmup(rounds=1)

    def test_verify_passes(self):
        node = StubDetector()
        node.load()
        assert node.verify() is True

    def test_verify_fails_bad_output(self):
        node = StubDetectorBadOutput()
        node.load()
        assert node.verify() is False
        assert node.state == NodeState.FAILED

    def test_repr(self):
        node = StubDetector(node_id="my_det")
        r = repr(node)
        assert "my_det" in r
        assert "unloaded" in r


# ===================================================================
# 4. Training Tests
# ===================================================================

class TestModelNodeTraining:
    """Test async training interface."""

    def test_train_not_implemented(self):
        node = StubDetector()
        node.load()
        with pytest.raises(NotImplementedError):
            asyncio.run(node.train("dataset.yaml", {"epochs": 1}))

    def test_train_with_impl(self):
        train_log = []

        class TrainableNode(StubDetector):
            async def _train_impl(self, dataset_ref, config):
                train_log.append({"dataset": dataset_ref, **config})

        node = TrainableNode()
        node.load()
        asyncio.run(node.train("my_data.yaml", {"epochs": 5}))
        assert node.state == NodeState.READY
        assert train_log[0]["epochs"] == 5

    def test_train_failure_sets_failed_state(self):
        class FailingNode(StubDetector):
            async def _train_impl(self, dataset_ref, config):
                raise RuntimeError("GPU OOM")

        node = FailingNode()
        node.load()
        with pytest.raises(RuntimeError, match="GPU OOM"):
            asyncio.run(node.train("data.yaml"))
        assert node.state == NodeState.FAILED


# ===================================================================
# 5. ModelResolver Tests
# ===================================================================

class TestModelResolver:
    """Test model variant resolution."""

    def test_fallback_only(self):
        resolver = ModelResolver(gpu_vram_gb=0.0)
        result = resolver.resolve(variants=None, fallback="yolov8n")
        assert result == "yolov8n"

    def test_no_variants_no_fallback_raises(self):
        resolver = ModelResolver(gpu_vram_gb=0.0)
        with pytest.raises(ValueError, match="No 'models' variants"):
            resolver.resolve(variants=None, fallback=None)

    def test_explicit_mode(self):
        resolver = ModelResolver(gpu_vram_gb=8.0)
        variants = {
            "speed": {"id": "yolov8n", "min_vram_gb": 0},
            "accuracy": {"id": "yolov8x", "min_vram_gb": 8},
        }
        result = resolver.resolve(variants, mode="speed")
        assert result == "yolov8n"

    def test_auto_selects_best_fit(self):
        resolver = ModelResolver(gpu_vram_gb=6.0)
        variants = {
            "speed": {"id": "yolov8n", "min_vram_gb": 0},
            "balanced": {"id": "yolov8m", "min_vram_gb": 4},
            "accuracy": {"id": "yolov8x", "min_vram_gb": 8},
        }
        result = resolver.resolve(variants, mode="auto")
        # 6GB fits balanced (4GB min) but not accuracy (8GB min)
        assert result == "yolov8m"

    def test_auto_low_vram_picks_speed(self):
        resolver = ModelResolver(gpu_vram_gb=1.0)
        variants = {
            "speed": {"id": "yolov8n", "min_vram_gb": 0},
            "accuracy": {"id": "yolov8x", "min_vram_gb": 8},
        }
        result = resolver.resolve(variants, mode="auto")
        assert result == "yolov8n"


# ===================================================================
# 6. ModelCache Tests
# ===================================================================

class TestModelCache:
    """Test model caching and download."""

    def test_existing_file_returned_directly(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"fake model")
            f.flush()
            cache = ModelCache()
            result = cache.get_or_download(f.name)
            assert result == f.name
            os.unlink(f.name)

    def test_passthrough_for_library_ids(self):
        cache = ModelCache()
        result = cache.get_or_download("google/siglip2-base-patch16-224")
        assert result == "google/siglip2-base-patch16-224"

    def test_download_called_when_not_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            def fake_download(model_id, dest):
                dest.write_text("downloaded")
                return dest

            result = cache.get_or_download("my_model.pt", downloader=fake_download)
            assert os.path.exists(result)
            assert Path(result).read_text() == "downloaded"

    def test_cached_file_returned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)
            # Pre-populate cache
            cached = Path(tmpdir) / "model.pt"
            cached.write_text("cached")

            result = cache.get_or_download("model.pt")
            assert result == str(cached)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)
            (Path(tmpdir) / "model.pt").write_text("data")
            assert len(list(Path(tmpdir).iterdir())) == 1
            cache.clear()
            assert len(list(Path(tmpdir).iterdir())) == 0

    def test_root_cache_env_used_when_no_constructor_override(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("VISION_TOOLS_CACHE_DIR", tmpdir)
            cache = ModelCache()
            assert cache.cache_dir == Path(tmpdir) / "models"

    def test_bare_checkpoint_id_resolves_to_cache_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)
            result = cache.get_or_download("yoloe-11s-seg.pt")
            assert result == str(Path(tmpdir) / "yoloe-11s-seg.pt")

    def test_bare_checkpoint_id_with_unimplemented_downloader_resolves_to_cache_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ModelCache(cache_dir=tmpdir)

            def unimplemented_download(_model_id, _dest):
                raise NotImplementedError()

            result = cache.get_or_download(
                "yoloe-11s-seg.pt",
                downloader=unimplemented_download,
            )
            assert result == str(Path(tmpdir) / "yoloe-11s-seg.pt")
