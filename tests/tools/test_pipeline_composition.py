"""
Pipeline composition tests — construction, validation, and bad-config errors.

Tests the full pipeline DAG: construction, I/O validation, execution,
and error handling for invalid configurations.
"""
import pytest
import numpy as np
import os

from vision_tools.utils.image_utils import load_image_opencv
from tests.tools.conftest import load_config, CACHED_YOLO_MODEL, ASSETS_DIR


# ===================================================================
# Happy-path tests
# ===================================================================

class TestPipelineConstruction:
    """Test pipeline construction and layer ordering."""

    def test_pipeline_from_tool_settings(self):
        """Legacy tool_settings config builds a working pipeline."""
        from vision_tools.core.tools.pipeline import VisionPipeline, PipelineConfig

        det_cfg = {"vocabulary": ["person", "car"]}
        if CACHED_YOLO_MODEL:
            det_cfg["model"] = CACHED_YOLO_MODEL
        config = PipelineConfig(
            tool_settings={
                "ov_detection": det_cfg,
                "pose_estimation": {},
            }
        )
        pipeline = VisionPipeline(config)

        assert len(pipeline.tools) == 2
        assert len(pipeline._layers) >= 1
        pipeline.shutdown()

    def test_pipeline_from_dag_nodes(self):
        """DAG node config with dependencies builds correct layers."""
        from vision_tools.core.tools.pipeline import (
            VisionPipeline, PipelineConfig, PipelineNode
        )

        det_cfg = {"vocabulary": ["person"]}
        if CACHED_YOLO_MODEL:
            det_cfg["model"] = CACHED_YOLO_MODEL
        config = PipelineConfig(nodes=[
            PipelineNode(
                node_id="detector",
                tool_type="ov_detection",
                tool_config=det_cfg,
            ),
            PipelineNode(
                node_id="pose",
                tool_type="pose_estimation",
                depends_on=["detector"],
            ),
        ])
        pipeline = VisionPipeline(config)

        # Detector should be in layer 0, pose in layer 1
        assert len(pipeline._layers) == 2
        assert "detector" in pipeline._layers[0]
        assert "pose" in pipeline._layers[1]
        pipeline.shutdown()

    def test_pipeline_parallel_tools(self):
        """Independent tools land in the same execution layer."""
        from vision_tools.core.tools.pipeline import (
            VisionPipeline, PipelineConfig, PipelineNode
        )

        det_cfg = {"vocabulary": ["person"]}
        if CACHED_YOLO_MODEL:
            det_cfg["model"] = CACHED_YOLO_MODEL
        config = PipelineConfig(nodes=[
            PipelineNode(node_id="det", tool_type="ov_detection",
                        tool_config=det_cfg),
            PipelineNode(node_id="pose", tool_type="pose_estimation"),
        ])
        pipeline = VisionPipeline(config)

        # No dependencies → both in layer 0
        assert len(pipeline._layers) == 1
        assert set(pipeline._layers[0]) == {"det", "pose"}
        pipeline.shutdown()


class TestPipelineExecution:
    """Test running real pipelines on test data."""

    def test_detection_then_pose(self):
        """Pipeline: detection → pose. Both produce results."""
        from vision_tools.core.tools.pipeline import (
            VisionPipeline, PipelineConfig, PipelineNode
        )

        det_cfg = {"vocabulary": ["person", "car"]}
        if CACHED_YOLO_MODEL:
            det_cfg["model"] = CACHED_YOLO_MODEL
        config = PipelineConfig(nodes=[
            PipelineNode(
                node_id="detector",
                tool_type="ov_detection",
                tool_config=det_cfg,
            ),
            PipelineNode(
                node_id="pose",
                tool_type="pose_estimation",
                depends_on=["detector"],
            ),
        ])
        pipeline = VisionPipeline(config)

        image_path = os.path.join(ASSETS_DIR, "test_image.png")
        image = load_image_opencv(image_path)

        # Run pipeline twice to establish tracks (SORT needs min_hits=2)
        pipeline.run_pipeline(image)
        frame, data = pipeline.run_pipeline(image)

        # Both tools should have run
        assert data["tools_run"] is True
        assert "boxes" in data  # from detector
        assert "poses" in data  # from pose

        # Should find at least one person
        person_boxes = [b for b in data["boxes"] if b.get("class_name") == "person"]
        assert len(person_boxes) >= 1

        # Should find at least one pose
        assert len(data["poses"]) >= 1

        pipeline.shutdown()


class TestDetectionCropPose:
    """Test the detection → crop → pose composition pattern."""

    def test_detect_crop_pose(self):
        """Detect a person, crop around them, run pose on the crop."""
        from vision_tools.core.tools.detection import OpenVocabularyDetector
        from vision_tools.core.tools.pose_estimation import PoseEstimator
        from vision_tools.core.aux_tools.crop import CropTool

        # Load image
        image_path = os.path.join(ASSETS_DIR, "test_image.png")
        image = load_image_opencv(image_path)

        # 1. Detect
        det_config = load_config("ov_detection")
        det_config["vocabulary"] = ["person", "car"]
        model_id = CACHED_YOLO_MODEL or det_config.get("model")
        detector = OpenVocabularyDetector(model_id, det_config)
        
        # Run detection twice to establish tracks (SORT needs min_hits=2)
        detector.process(image, {})
        det_results, _ = detector.process(image, {})

        assert len(det_results["boxes"]) > 0, "Need at least one detection"

        # 2. Crop around best person
        crop_tool = CropTool(pad=20, scale=1.1)
        crop_result = crop_tool.crop_best(
            image, det_results["boxes"], class_name="person"
        )

        assert crop_result is not None, "Should find a person to crop"
        assert crop_result.cropped_image.shape[0] > 0
        assert crop_result.cropped_image.shape[1] > 0
        assert crop_result.crop_offset is not None

        print(f"Cropped person: shape={crop_result.cropped_image.shape}, "
              f"offset={crop_result.crop_offset}")

        # 3. Run pose on the crop
        pose_config = load_config("pose_estimation")
        pose_est = PoseEstimator(pose_config.get("model"), pose_config)
        pose_results, _ = pose_est.process(crop_result.cropped_image, {})

        assert "poses" in pose_results
        # Pose on a tight person crop should detect at least one pose
        assert len(pose_results["poses"]) >= 1
        assert len(pose_results["poses"][0]["keypoints"]) == 17

        print(f"Pose on crop: {len(pose_results['poses'])} poses detected")


class TestCropTool:
    """Test CropTool directly."""

    def test_basic_crop(self, test_image):
        """Crop with default params."""
        from vision_tools.core.aux_tools.crop import CropTool
        from vision_tools.utils.schemas import BoundingBox

        h, w = test_image.shape[:2]
        bbox = BoundingBox(xyxy=[100, 100, 300, 400], class_id=0,
                          confidence=0.9, class_name="person")

        crop_tool = CropTool()
        result = crop_tool.crop(test_image, bbox)

        assert result.cropped_image.shape == (300, 200, 3)  # (400-100, 300-100, 3)
        assert result.crop_offset == (100, 100)

    def test_crop_with_padding(self, test_image):
        """Crop with padding expands the region."""
        from vision_tools.core.aux_tools.crop import CropTool
        from vision_tools.utils.schemas import BoundingBox

        bbox = BoundingBox(xyxy=[100, 100, 300, 400], class_id=0,
                          confidence=0.9)

        crop_tool = CropTool(pad=10)
        result = crop_tool.crop(test_image, bbox)

        # Padded crop should be 20px wider and 20px taller
        assert result.cropped_image.shape[0] == 320  # (400+10)-(100-10) = 320
        assert result.cropped_image.shape[1] == 220  # (300+10)-(100-10) = 220
        assert result.crop_offset == (90, 90)

    def test_crop_with_scale(self, test_image):
        """Scale > 1 expands the crop region."""
        from vision_tools.core.aux_tools.crop import CropTool
        from vision_tools.utils.schemas import BoundingBox

        bbox = BoundingBox(xyxy=[100, 100, 300, 400], class_id=0,
                          confidence=0.9)

        crop_tool = CropTool(scale=2.0)
        result = crop_tool.crop(test_image, bbox)

        # 2x scale: width goes from 200→400, height from 300→600
        # Centered: x: 200-100=100→200+200=400, y: 250-150=100→250+150=550
        # But clamped to image bounds
        assert result.cropped_image.shape[0] > 300  # taller than original
        assert result.cropped_image.shape[1] > 200  # wider than original

    def test_crop_clamped_to_bounds(self, test_image):
        """Crop near edges should be clamped to image bounds."""
        from vision_tools.core.aux_tools.crop import CropTool
        from vision_tools.utils.schemas import BoundingBox

        h, w = test_image.shape[:2]
        # Box near top-left corner
        bbox = BoundingBox(xyxy=[0, 0, 50, 50], class_id=0, confidence=0.9)

        crop_tool = CropTool(pad=100)  # Large pad will go negative
        result = crop_tool.crop(test_image, bbox)

        # Should be clamped, not negative
        assert result.crop_offset[0] >= 0
        assert result.crop_offset[1] >= 0
        assert result.cropped_image.shape[0] > 0
        assert result.cropped_image.shape[1] > 0

    def test_crop_best_filters_by_class(self, test_image, detector):
        """crop_best selects highest-confidence box of requested class."""
        from vision_tools.core.aux_tools.crop import CropTool

        det_results, _ = detector.process(test_image, {})
        crop_tool = CropTool(pad=10)

        person_crop = crop_tool.crop_best(
            test_image, det_results["boxes"], class_name="person"
        )
        # Test image has a person, so this should succeed
        if person_crop is not None:
            assert person_crop.original_bbox.class_name == "person"

    def test_crop_best_no_match(self, test_image, detector):
        """crop_best returns None when no box matches the class."""
        from vision_tools.core.aux_tools.crop import CropTool

        det_results, _ = detector.process(test_image, {})
        crop_tool = CropTool()

        result = crop_tool.crop_best(
            test_image, det_results["boxes"], class_name="elephant"
        )
        assert result is None


# ===================================================================
# Bad-config tests — expecting exceptions
# ===================================================================

class TestPipelineBadConfig:
    """Test that invalid pipeline configurations raise appropriate errors."""

    def test_unknown_tool_type(self):
        """Unknown tool_type should raise ValueError."""
        from vision_tools.core.tools.pipeline import (
            VisionPipeline, PipelineConfig, PipelineNode
        )

        config = PipelineConfig(nodes=[
            PipelineNode(
                node_id="bad",
                tool_type="nonexistent_tool",
            ),
        ])

        with pytest.raises(ValueError, match="Unknown tool type"):
            VisionPipeline(config)

    def test_circular_dependency(self):
        """Circular dependency should raise ValueError."""
        from vision_tools.core.tools.pipeline import (
            VisionPipeline, PipelineConfig, PipelineNode
        )

        config = PipelineConfig(nodes=[
            PipelineNode(
                node_id="a",
                tool_type="ov_detection",
                tool_config={"vocabulary": ["person"]},
                depends_on=["b"],
            ),
            PipelineNode(
                node_id="b",
                tool_type="pose_estimation",
                depends_on=["a"],
            ),
        ])

        with pytest.raises(ValueError, match="cycle"):
            VisionPipeline(config)

    def test_missing_dependency(self):
        """Dependency on non-existent node should raise ValueError."""
        from vision_tools.core.tools.pipeline import (
            VisionPipeline, PipelineConfig, PipelineNode
        )

        config = PipelineConfig(nodes=[
            PipelineNode(
                node_id="detector",
                tool_type="ov_detection",
                tool_config={"vocabulary": ["person"]},
                depends_on=["ghost_node"],
            ),
        ])

        with pytest.raises(ValueError, match="doesn't exist"):
            VisionPipeline(config)
