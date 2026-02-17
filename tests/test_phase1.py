"""
Phase 1 Basic Functionality Tests
==================================

Tests for the new Phase 1 infrastructure without loading heavy ML models.
Covers: schemas, training config/dataset, state manager, dynamic batcher.

Run: pytest tests/test_phase1.py -v
"""
import asyncio
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pydantic import ValidationError


# ===================================================================
# 1. Schema Tests
# ===================================================================

class TestToolState:
    """Test ToolState enum and transitions."""

    def test_valid_transitions(self):
        from vision_tools.utils.schemas import ToolState
        # BASE can go to TRAINING or TUNED
        assert ToolState.BASE.can_transition_to(ToolState.TRAINING)
        assert ToolState.BASE.can_transition_to(ToolState.TUNED)
        # TRAINING can go to TUNED, BASE, UNTRAINED (rollback)
        assert ToolState.TRAINING.can_transition_to(ToolState.TUNED)
        assert ToolState.TRAINING.can_transition_to(ToolState.BASE)
        assert ToolState.TRAINING.can_transition_to(ToolState.UNTRAINED)
        # TUNED can re-train
        assert ToolState.TUNED.can_transition_to(ToolState.TRAINING)
        # UNTRAINED can start training
        assert ToolState.UNTRAINED.can_transition_to(ToolState.TRAINING)

    def test_invalid_transitions(self):
        from vision_tools.utils.schemas import ToolState
        # UNTRAINED cannot skip to TUNED or BASE
        assert not ToolState.UNTRAINED.can_transition_to(ToolState.TUNED)
        assert not ToolState.UNTRAINED.can_transition_to(ToolState.BASE)
        # BASE cannot go directly to UNTRAINED
        assert not ToolState.BASE.can_transition_to(ToolState.UNTRAINED)
        # TUNED cannot go to BASE or UNTRAINED directly
        assert not ToolState.TUNED.can_transition_to(ToolState.BASE)
        assert not ToolState.TUNED.can_transition_to(ToolState.UNTRAINED)


class TestBoundingBox:
    """Test BoundingBox schema validation."""

    def test_valid_bbox(self):
        from vision_tools.utils.schemas import BoundingBox
        bb = BoundingBox(xyxy=[10, 20, 100, 200], class_id=0, confidence=0.95,
                         class_name="person")
        assert bb.xyxy == [10, 20, 100, 200]
        assert bb.class_id == 0
        assert bb.confidence == 0.95
        assert bb.class_name == "person"
        assert bb.tracker_id is None

    def test_bbox_with_tracker(self):
        from vision_tools.utils.schemas import BoundingBox
        bb = BoundingBox(xyxy=[0, 0, 50, 50], class_id=1, confidence=0.8,
                         tracker_id=42)
        assert bb.tracker_id == 42

    def test_bbox_invalid_confidence(self):
        from vision_tools.utils.schemas import BoundingBox
        with pytest.raises(ValidationError):
            BoundingBox(xyxy=[0, 0, 1, 1], class_id=0, confidence=1.5)

    def test_bbox_invalid_xyxy_length(self):
        from vision_tools.utils.schemas import BoundingBox
        with pytest.raises(ValidationError):
            BoundingBox(xyxy=[0, 0, 1], class_id=0, confidence=0.5)

    def test_bbox_serialization(self):
        from vision_tools.utils.schemas import BoundingBox
        bb = BoundingBox(xyxy=[1, 2, 3, 4], class_id=5, confidence=0.7,
                         class_name="car")
        d = bb.model_dump()
        assert d["xyxy"] == [1, 2, 3, 4]
        assert d["class_id"] == 5
        restored = BoundingBox.model_validate(d)
        assert restored == bb


class TestDetectionResult:
    """Test DetectionResult schema."""

    def test_empty_result(self):
        from vision_tools.utils.schemas import DetectionResult
        dr = DetectionResult()
        assert dr.boxes == []
        assert dr.class_names is None

    def test_result_with_boxes(self):
        from vision_tools.utils.schemas import BoundingBox, DetectionResult
        boxes = [
            BoundingBox(xyxy=[0, 0, 10, 10], class_id=0, confidence=0.9),
            BoundingBox(xyxy=[20, 20, 30, 30], class_id=1, confidence=0.8),
        ]
        dr = DetectionResult(boxes=boxes, class_names={0: "person", 1: "car"})
        assert len(dr.boxes) == 2
        assert dr.class_names[0] == "person"


class TestSegmentationMask:
    """Test SegmentationMask RLE encode/decode round-trip."""

    def test_rle_round_trip(self):
        from vision_tools.utils.schemas import SegmentationMask
        # Create a 5x5 mask with a 3x3 filled center region
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 1

        encoded = SegmentationMask.from_binary_mask(mask, class_id=0,
                                                     confidence=0.9,
                                                     class_name="test")
        decoded = encoded.to_binary_mask()
        np.testing.assert_array_equal(mask, decoded)

    def test_empty_mask(self):
        from vision_tools.utils.schemas import SegmentationMask
        mask = np.zeros((4, 4), dtype=np.uint8)
        encoded = SegmentationMask.from_binary_mask(mask, class_id=0,
                                                     confidence=0.5)
        decoded = encoded.to_binary_mask()
        np.testing.assert_array_equal(mask, decoded)


class TestEmbedding:
    """Test Embedding and EmbeddingResult schemas."""

    def test_valid_embedding(self):
        from vision_tools.utils.schemas import Embedding, EmbeddingResult
        emb = Embedding(vector=[0.1, 0.2, 0.3], model_id="test-model",
                        dimension=3)
        result = EmbeddingResult(embedding=emb)
        assert result.embedding.dimension == 3
        d = result.model_dump()
        assert d["embedding"]["vector"] == [0.1, 0.2, 0.3]

    def test_invalid_dimension(self):
        from vision_tools.utils.schemas import Embedding
        with pytest.raises(ValidationError):
            Embedding(vector=[0.1], model_id="test", dimension=0)


class TestPoseSchemas:
    """Test Keypoint, PoseKeypoints, PoseResult."""

    def test_keypoint_creation(self):
        from vision_tools.utils.schemas import Keypoint
        kp = Keypoint(x=100.0, y=200.0, confidence=0.95)
        assert kp.x == 100.0
        assert kp.confidence == 0.95

    def test_pose_result(self):
        from vision_tools.utils.schemas import Keypoint, PoseKeypoints, PoseResult
        kpts = [Keypoint(x=float(i), y=float(i*2), confidence=0.9) for i in range(17)]
        pose = PoseKeypoints(person_id=0, keypoints=kpts)
        result = PoseResult(poses=[pose])
        assert len(result.poses) == 1
        assert len(result.poses[0].keypoints) == 17
        d = result.model_dump()
        assert len(d["poses"][0]["keypoints"]) == 17


class TestCaptionSchemas:
    """Test Caption and CaptionResult."""

    def test_caption_result(self):
        from vision_tools.utils.schemas import Caption, CaptionResult
        cap = Caption(text="A dog running in a park", model_id="smolvlm2")
        result = CaptionResult(caption=cap)
        assert result.caption.text == "A dog running in a park"
        d = result.model_dump()
        assert d["caption"]["text"] == "A dog running in a park"


class TestFrameMetadata:
    """Test FrameMetadata schema."""

    def test_defaults(self):
        from vision_tools.utils.schemas import FrameMetadata
        fm = FrameMetadata(frame_idx=0)
        assert fm.timestamp == 0.0
        assert fm.scene_change_score == 0.0
        assert fm.camera_id == "default"

    def test_validation(self):
        from vision_tools.utils.schemas import FrameMetadata
        with pytest.raises(ValidationError):
            FrameMetadata(frame_idx=-1)  # ge=0


class TestFrameResult:
    """Test FrameResult container."""

    def test_empty_result(self):
        from vision_tools.utils.schemas import FrameMetadata, FrameResult
        fm = FrameMetadata(frame_idx=0, timestamp=0.5)
        fr = FrameResult(metadata=fm)
        assert fr.results == {}
        assert fr.tools_run is False

    def test_result_with_data(self):
        from vision_tools.utils.schemas import FrameMetadata, FrameResult
        fm = FrameMetadata(frame_idx=1, timestamp=1.0, scene_change_score=0.3)
        fr = FrameResult(
            metadata=fm,
            results={"detector": {"boxes": [{"xyxy": [1,2,3,4]}]}},
            tools_run=True,
        )
        assert fr.tools_run
        assert "detector" in fr.results


class TestBatchPayload:
    """Test BatchPayload container."""

    def test_batch_creation(self):
        from vision_tools.utils.schemas import FrameMetadata, BatchPayload
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        metadatas = [FrameMetadata(frame_idx=i, timestamp=i*0.033) for i in range(4)]
        bp = BatchPayload(batch_id="test-001", frames=frames,
                         frame_metadatas=metadatas)
        assert len(bp) == 4
        assert bp.batch_id == "test-001"
        assert bp.frame_results == []

    def test_empty_batch(self):
        from vision_tools.utils.schemas import BatchPayload
        bp = BatchPayload(batch_id="empty")
        assert len(bp) == 0


# ===================================================================
# 2. Training Config Tests
# ===================================================================

class TestTrainConfig:
    """Test TrainConfig validation and framework kwargs conversion."""

    def test_defaults(self):
        from vision_tools.training.config import TrainConfig
        tc = TrainConfig()
        assert tc.epochs == 50
        assert tc.batch_size == 16
        assert tc.learning_rate == 0.01
        assert tc.imgsz == 640

    def test_custom_config(self):
        from vision_tools.training.config import TrainConfig
        tc = TrainConfig(epochs=100, batch_size=32, learning_rate=0.001,
                        imgsz=1280)
        assert tc.epochs == 100
        assert tc.batch_size == 32
        assert tc.learning_rate == 0.001

    def test_validation_bounds(self):
        from vision_tools.training.config import TrainConfig
        with pytest.raises(ValidationError):
            TrainConfig(epochs=0)  # ge=1
        with pytest.raises(ValidationError):
            TrainConfig(learning_rate=0.0)  # gt=0
        with pytest.raises(ValidationError):
            TrainConfig(val_split=0.6)  # le=0.5

    def test_to_framework_kwargs(self):
        from vision_tools.training.config import TrainConfig
        tc = TrainConfig(epochs=10, batch_size=8, learning_rate=0.005,
                        device="cuda", checkpoint_dir="/tmp/checkpoints",
                        extra_args={"half": True})
        kwargs = tc.to_framework_kwargs()
        assert kwargs["epochs"] == 10
        assert kwargs["batch"] == 8
        assert kwargs["lr0"] == 0.005
        assert kwargs["device"] == "cuda"
        assert kwargs["project"] == "/tmp/checkpoints"
        assert kwargs["half"] is True

    def test_to_framework_kwargs_no_optionals(self):
        from vision_tools.training.config import TrainConfig
        tc = TrainConfig(epochs=20)
        kwargs = tc.to_framework_kwargs()
        assert "device" not in kwargs
        assert "project" not in kwargs

    def test_augmentation_preset(self):
        from vision_tools.training.config import TrainConfig, AugmentationPreset
        tc = TrainConfig(augmentation=AugmentationPreset.HEAVY)
        assert tc.augmentation == AugmentationPreset.HEAVY

    def test_serialization(self):
        from vision_tools.training.config import TrainConfig
        tc = TrainConfig(epochs=25, batch_size=4)
        d = tc.model_dump()
        restored = TrainConfig.model_validate(d)
        assert restored.epochs == 25
        assert restored.batch_size == 4


# ===================================================================
# 3. VisionDataset Tests
# ===================================================================

class TestVisionDataset:
    """Test VisionDataset format detection and metadata."""

    def test_auto_detect_coco_json(self):
        from vision_tools.training.dataset import VisionDataset, DatasetFormat
        ds = VisionDataset("/fake/path/annotations.json")
        assert ds.format == DatasetFormat.COCO

    def test_auto_detect_yolo_yaml(self):
        from vision_tools.training.dataset import VisionDataset, DatasetFormat
        ds = VisionDataset("/fake/path/data.yaml")
        assert ds.format == DatasetFormat.YOLO

    def test_explicit_format(self):
        from vision_tools.training.dataset import VisionDataset, DatasetFormat
        ds = VisionDataset("/fake/path", format=DatasetFormat.CUSTOM)
        assert ds.format == DatasetFormat.CUSTOM

    def test_name_default(self):
        from vision_tools.training.dataset import VisionDataset
        ds = VisionDataset("/fake/path/my_dataset.yaml")
        assert ds.name == "my_dataset"

    def test_name_override(self):
        from vision_tools.training.dataset import VisionDataset
        ds = VisionDataset("/fake/path", name="Custom Name")
        assert ds.name == "Custom Name"

    def test_repr(self):
        from vision_tools.training.dataset import VisionDataset
        ds = VisionDataset("/fake/path/data.yaml", name="test")
        # Pre-populate to avoid lazy-load hitting filesystem
        ds._loaded = True
        ds._splits = {}
        r = repr(ds)
        assert "test" in r
        assert "yolo" in r

    def test_validate_for_tool_empty_dataset(self):
        from vision_tools.training.dataset import VisionDataset
        ds = VisionDataset("/fake/path/data.yaml", name="test")
        # Force loaded state with no images
        ds._loaded = True
        ds._splits = {}
        errors = ds.validate_for_tool("ov_detection")
        assert any("no images" in e.lower() for e in errors)

    def test_get_ref_yolo(self):
        from vision_tools.training.dataset import VisionDataset, DatasetFormat
        ds = VisionDataset("/fake/path/data.yaml")
        ds._loaded = True
        ds._metadata = {"yaml_path": "/fake/path/data.yaml"}
        assert ds.get_ref() == "/fake/path/data.yaml"

    def test_get_ref_coco(self):
        from vision_tools.training.dataset import VisionDataset, DatasetFormat
        ds = VisionDataset("/fake/path/annotations.json")
        assert ds.get_ref() == "/fake/path/annotations.json"


# ===================================================================
# 4. State Manager Tests
# ===================================================================

class TestStateManager:
    """Test StateManager rule registration, evaluation, and alerts."""

    def _make_frame_result(self, frame_idx, timestamp, detections=None):
        from vision_tools.utils.schemas import FrameMetadata, FrameResult
        fm = FrameMetadata(frame_idx=frame_idx, timestamp=timestamp)
        results = {}
        if detections:
            results["boxes"] = detections
        return FrameResult(metadata=fm, results=results, tools_run=True)

    def test_register_and_list_rules(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        rule = StateRule(rule_id="test", rule_type=RuleType.OBJECT_PRESENCE,
                        params={"class_name": "person"})
        sm.register_rule(rule)
        assert "test" in sm.rules
        assert sm.rules["test"].rule_id == "test"

    def test_unregister_rule(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(rule_id="r1", rule_type=RuleType.OBJECT_PRESENCE))
        sm.unregister_rule("r1")
        assert "r1" not in sm.rules

    def test_object_presence_triggers(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="person_alert",
            rule_type=RuleType.OBJECT_PRESENCE,
            params={"class_name": "person", "confidence_threshold": 0.5}
        ))
        fr = self._make_frame_result(0, 1.0, [
            {"xyxy": [10, 20, 100, 200], "class_name": "person", "confidence": 0.9}
        ])
        alerts = sm.evaluate([fr])
        assert len(alerts) == 1
        assert alerts[0].rule_id == "person_alert"
        assert "person" in alerts[0].message

    def test_object_presence_no_trigger_low_conf(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="person_alert",
            rule_type=RuleType.OBJECT_PRESENCE,
            params={"class_name": "person", "confidence_threshold": 0.8}
        ))
        fr = self._make_frame_result(0, 1.0, [
            {"xyxy": [0, 0, 1, 1], "class_name": "person", "confidence": 0.3}
        ])
        alerts = sm.evaluate([fr])
        assert len(alerts) == 0

    def test_object_presence_no_trigger_wrong_class(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="person_alert",
            rule_type=RuleType.OBJECT_PRESENCE,
            params={"class_name": "person", "confidence_threshold": 0.5}
        ))
        fr = self._make_frame_result(0, 1.0, [
            {"xyxy": [0, 0, 1, 1], "class_name": "car", "confidence": 0.9}
        ])
        alerts = sm.evaluate([fr])
        assert len(alerts) == 0

    def test_object_absence_triggers(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="absence_alert",
            rule_type=RuleType.OBJECT_ABSENCE,
            params={"class_name": "person", "seconds": 2.0}
        ))
        # Frame at t=0 has person, then frames at t=1,2,3 without
        frames = [
            self._make_frame_result(0, 0.0, [
                {"xyxy": [0,0,1,1], "class_name": "person", "confidence": 0.9}
            ]),
            self._make_frame_result(1, 1.0, []),
            self._make_frame_result(2, 2.0, []),
            self._make_frame_result(3, 3.0, []),
        ]
        alerts = sm.evaluate(frames)
        # Person was last seen at t=0, latest is t=3 → 3s absence > 2s threshold
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert "absent" in alerts[0].message

    def test_object_absence_no_trigger(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="absence_alert",
            rule_type=RuleType.OBJECT_ABSENCE,
            params={"class_name": "person", "seconds": 5.0}
        ))
        frames = [
            self._make_frame_result(0, 0.0, [
                {"xyxy": [0,0,1,1], "class_name": "person", "confidence": 0.9}
            ]),
            self._make_frame_result(1, 1.0, []),
        ]
        alerts = sm.evaluate(frames)
        # Only 1s absence < 5s threshold
        assert len(alerts) == 0

    def test_count_threshold_triggers(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="crowd_alert",
            rule_type=RuleType.COUNT_THRESHOLD,
            params={"class_name": "person", "max_count": 2}
        ))
        detections = [
            {"xyxy": [i*10, 0, i*10+10, 10], "class_name": "person", "confidence": 0.9}
            for i in range(5)
        ]
        fr = self._make_frame_result(0, 1.0, detections)
        alerts = sm.evaluate([fr])
        assert len(alerts) == 1
        assert "5 > 2" in alerts[0].message

    def test_count_threshold_no_trigger(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="crowd_alert",
            rule_type=RuleType.COUNT_THRESHOLD,
            params={"class_name": "person", "max_count": 10}
        ))
        fr = self._make_frame_result(0, 1.0, [
            {"xyxy": [0,0,1,1], "class_name": "person", "confidence": 0.9}
        ])
        alerts = sm.evaluate([fr])
        assert len(alerts) == 0

    def test_zone_entry_triggers(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="zone_alert",
            rule_type=RuleType.ZONE_ENTRY,
            params={
                # Normalized coords (0-1) matching ZoneDefinition constraints
                "zone": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0, "name": "entrance"},
                "class_name": "person",
            }
        ))
        # Object centered at (50, 50) → inside normalized zone [0, 1]
        fr = self._make_frame_result(0, 1.0, [
            {"xyxy": [0.3, 0.3, 0.7, 0.7], "class_name": "person", "confidence": 0.9}
        ])
        alerts = sm.evaluate([fr])
        assert len(alerts) == 1
        assert "entrance" in alerts[0].message

    def test_zone_entry_no_trigger_outside(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="zone_alert",
            rule_type=RuleType.ZONE_ENTRY,
            params={
                "zone": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5, "name": "corner"},
            }
        ))
        # Object centered at (0.75, 0.75) → outside zone [0, 0.5]
        fr = self._make_frame_result(0, 1.0, [
            {"xyxy": [0.5, 0.5, 1.0, 1.0], "class_name": "person", "confidence": 0.9}
        ])
        alerts = sm.evaluate([fr])
        assert len(alerts) == 0

    def test_custom_evaluator(self):
        from vision_tools.core.state_manager import (
            StateManager, StateRule, RuleType, Alert
        )
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="custom_test",
            rule_type=RuleType.CUSTOM,
            params={"threshold": 42}
        ))

        def my_evaluator(history, params):
            return [Alert(
                rule_id="custom_test",
                rule_type=RuleType.CUSTOM,
                timestamp=0.0,
                frame_idx=0,
                message=f"Custom alert with threshold={params.get('threshold')}",
            )]

        sm.register_custom_evaluator("custom_test", my_evaluator)
        fr = self._make_frame_result(0, 0.0)
        alerts = sm.evaluate([fr])
        assert len(alerts) == 1
        assert "threshold=42" in alerts[0].message

    def test_disabled_rule_skipped(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="disabled",
            rule_type=RuleType.OBJECT_PRESENCE,
            params={"class_name": "person"},
            enabled=False,
        ))
        fr = self._make_frame_result(0, 1.0, [
            {"xyxy": [0,0,1,1], "class_name": "person", "confidence": 0.9}
        ])
        alerts = sm.evaluate([fr])
        assert len(alerts) == 0

    def test_history_accumulation(self):
        from vision_tools.core.state_manager import StateManager
        sm = StateManager(history_size=5)
        frames = [self._make_frame_result(i, i * 0.1) for i in range(10)]
        sm.ingest(frames)
        assert sm.history_size == 5  # Capped at max

    def test_clear_history(self):
        from vision_tools.core.state_manager import StateManager
        sm = StateManager()
        frames = [self._make_frame_result(i, i * 0.1) for i in range(5)]
        sm.ingest(frames)
        assert sm.history_size == 5
        sm.clear_history()
        assert sm.history_size == 0

    def test_total_alerts_counter(self):
        from vision_tools.core.state_manager import StateManager, StateRule, RuleType
        sm = StateManager()
        sm.register_rule(StateRule(
            rule_id="alert",
            rule_type=RuleType.OBJECT_PRESENCE,
            params={"class_name": "person", "confidence_threshold": 0.5}
        ))
        fr = self._make_frame_result(0, 1.0, [
            {"xyxy": [0,0,1,1], "class_name": "person", "confidence": 0.9}
        ])
        sm.evaluate([fr])
        assert sm.total_alerts == 1
        sm.evaluate([fr])
        assert sm.total_alerts == 2


# ===================================================================
# 5. Dynamic Batcher Tests
# ===================================================================

class TestDynamicBatcher:
    """Test DynamicBatcher batching logic."""

    def _make_frame_and_meta(self, idx):
        from vision_tools.utils.schemas import FrameMetadata
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        meta = FrameMetadata(frame_idx=idx, timestamp=idx * 0.033)
        return frame, meta

    def test_full_batch_flush(self):
        """Batch flushes when max_batch_size is reached."""
        async def _run():
            from vision_tools.engine.batcher import DynamicBatcher
            from vision_tools.utils.schemas import BatchPayload

            batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=5000)
            in_q = asyncio.Queue()
            out_q = asyncio.Queue()

            for i in range(4):
                await in_q.put(self._make_frame_and_meta(i))
            await in_q.put(None)

            await batcher.run(in_q, out_q)

            batch = await out_q.get()
            assert isinstance(batch, BatchPayload)
            assert len(batch) == 4
            assert batch.frame_metadatas[0].frame_idx == 0
            assert batch.frame_metadatas[3].frame_idx == 3

            end = await out_q.get()
            assert end is None

        asyncio.run(_run())

    def test_partial_batch_on_eos(self):
        """Partial batch is flushed when end-of-stream arrives."""
        async def _run():
            from vision_tools.engine.batcher import DynamicBatcher
            from vision_tools.utils.schemas import BatchPayload

            batcher = DynamicBatcher(max_batch_size=10, max_wait_ms=5000)
            in_q = asyncio.Queue()
            out_q = asyncio.Queue()

            for i in range(3):
                await in_q.put(self._make_frame_and_meta(i))
            await in_q.put(None)

            await batcher.run(in_q, out_q)

            batch = await out_q.get()
            assert isinstance(batch, BatchPayload)
            assert len(batch) == 3

            end = await out_q.get()
            assert end is None

        asyncio.run(_run())

    def test_multiple_batches(self):
        """Multiple batches are emitted for > max_batch_size frames."""
        async def _run():
            from vision_tools.engine.batcher import DynamicBatcher

            batcher = DynamicBatcher(max_batch_size=3, max_wait_ms=5000)
            in_q = asyncio.Queue()
            out_q = asyncio.Queue()

            for i in range(7):
                await in_q.put(self._make_frame_and_meta(i))
            await in_q.put(None)

            await batcher.run(in_q, out_q)

            batches = []
            while True:
                item = await out_q.get()
                if item is None:
                    break
                batches.append(item)

            assert len(batches) == 3
            assert len(batches[0]) == 3
            assert len(batches[1]) == 3
            assert len(batches[2]) == 1

        asyncio.run(_run())

    def test_time_flush(self):
        """Partial batch is flushed after max_wait_ms timeout."""
        async def _run():
            from vision_tools.engine.batcher import DynamicBatcher
            from vision_tools.utils.schemas import BatchPayload

            batcher = DynamicBatcher(max_batch_size=100, max_wait_ms=50)
            in_q = asyncio.Queue()
            out_q = asyncio.Queue()

            for i in range(2):
                await in_q.put(self._make_frame_and_meta(i))

            async def delayed_eos():
                await asyncio.sleep(0.2)
                await in_q.put(None)

            asyncio.create_task(delayed_eos())
            await batcher.run(in_q, out_q)

            batch = await out_q.get()
            assert isinstance(batch, BatchPayload)
            assert len(batch) == 2

        asyncio.run(_run())

    def test_empty_stream(self):
        """Immediate end-of-stream produces no batches."""
        async def _run():
            from vision_tools.engine.batcher import DynamicBatcher

            batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100)
            in_q = asyncio.Queue()
            out_q = asyncio.Queue()

            await in_q.put(None)
            await batcher.run(in_q, out_q)

            end = await out_q.get()
            assert end is None

        asyncio.run(_run())

    def test_batch_ids_unique(self):
        """Each batch gets a unique batch_id."""
        async def _run():
            from vision_tools.engine.batcher import DynamicBatcher

            batcher = DynamicBatcher(max_batch_size=2, max_wait_ms=5000)
            in_q = asyncio.Queue()
            out_q = asyncio.Queue()

            for i in range(6):
                await in_q.put(self._make_frame_and_meta(i))
            await in_q.put(None)

            await batcher.run(in_q, out_q)

            ids = set()
            while True:
                item = await out_q.get()
                if item is None:
                    break
                ids.add(item.batch_id)

            assert len(ids) == 3  # All unique

        asyncio.run(_run())


# ===================================================================
# 6. ToolIOContract Tests
# ===================================================================

class TestToolIOContract:
    """Test ToolIOContract schema."""

    def test_contract_creation(self):
        from vision_tools.utils.schemas import ToolIOContract
        contract = ToolIOContract(
            key="boxes",
            schema_type="vision_tools.utils.schemas.DetectionResult",
            required=True,
            description="Detection output",
        )
        assert contract.key == "boxes"
        assert contract.required is True


# ===================================================================
# 7. ModelScale Tests
# ===================================================================

class TestModelScale:
    """Test ModelScale enum values."""

    def test_all_scales(self):
        from vision_tools.utils.schemas import ModelScale
        assert ModelScale.NANO == "nano"
        assert ModelScale.SMALL == "small"
        assert ModelScale.BASE == "base"
        assert ModelScale.LARGE == "large"
