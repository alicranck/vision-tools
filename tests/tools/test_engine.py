"""
Engine integration tests — VideoInferenceEngine with a synthetic test video.

Creates a short video from the test image, processes it through the engine,
and verifies FrameResults are produced.
"""
import asyncio
import os
import tempfile

import cv2
import numpy as np
import pytest

from vision_tools.utils.image_utils import load_image_opencv
from tests.tools.conftest import CACHED_YOLO_MODEL, ASSETS_DIR
NUM_TEST_FRAMES = 10


def _create_test_video(image: np.ndarray, num_frames: int = NUM_TEST_FRAMES) -> str:
    """Write N copies of the image as a .mp4 test video. Returns path."""
    h, w = image.shape[:2]
    path = os.path.join(tempfile.gettempdir(), "vision_tools_test.mp4")

    # OpenCV VideoWriter expects BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (w, h))

    for _ in range(num_frames):
        writer.write(bgr)
    writer.release()

    assert os.path.exists(path), f"Failed to create test video at {path}"
    return path


class TestEngine:
    """Test VideoInferenceEngine end-to-end."""

    def test_engine_processes_video(self):
        """Engine should process a synthetic video and return FrameResults."""
        from vision_tools.core.tools.pipeline import VisionPipeline, PipelineConfig
        from vision_tools.engine.video_engine import VideoInferenceEngine

        # Load test image and create video
        image_path = os.path.join(ASSETS_DIR, "test_image.png")
        image = load_image_opencv(image_path)
        video_path = _create_test_video(image, num_frames=6)

        # Build a minimal pipeline (detection only for speed)
        det_cfg = {"vocabulary": ["person", "car"]}
        if CACHED_YOLO_MODEL:
            det_cfg["model"] = CACHED_YOLO_MODEL
        config = PipelineConfig(
            tool_settings={"ov_detection": det_cfg}
        )
        pipeline = VisionPipeline(config)

        # Create engine and process
        engine = VideoInferenceEngine(
            tool_pipeline=pipeline,
            video_path=video_path,
            max_batch_size=3,
            max_wait_ms=100,
        )

        results = asyncio.run(engine.process_video())

        assert len(results) > 0, "Engine should produce at least one FrameResult"
        for fr in results:
            assert fr.metadata is not None
            assert fr.metadata.frame_idx >= 0

        print(f"Engine produced {len(results)} FrameResults from {6}-frame video")

        # Cleanup
        pipeline.shutdown()
        os.remove(video_path)

    def test_engine_batch_callback(self):
        """on_batch callback should fire with BatchPayload objects."""
        from vision_tools.core.tools.pipeline import VisionPipeline, PipelineConfig
        from vision_tools.engine.video_engine import VideoInferenceEngine
        from vision_tools.utils.schemas import BatchPayload

        image_path = os.path.join(ASSETS_DIR, "test_image.png")
        image = load_image_opencv(image_path)
        video_path = _create_test_video(image, num_frames=4)

        det_cfg = {"vocabulary": ["person", "car"]}
        if CACHED_YOLO_MODEL:
            det_cfg["model"] = CACHED_YOLO_MODEL
        config = PipelineConfig(
            tool_settings={"ov_detection": det_cfg}
        )
        pipeline = VisionPipeline(config)

        callback_payloads = []

        async def on_batch(payload):
            callback_payloads.append(payload)

        engine = VideoInferenceEngine(
            tool_pipeline=pipeline,
            video_path=video_path,
            max_batch_size=2,
            max_wait_ms=100,
        )

        asyncio.run(engine.process_video(on_batch=on_batch))

        assert len(callback_payloads) > 0, "on_batch callback should have fired"
        for payload in callback_payloads:
            assert hasattr(payload, "batch_id")
            assert hasattr(payload, "frame_results")

        print(f"Callback fired {len(callback_payloads)} times")

        pipeline.shutdown()
        os.remove(video_path)
