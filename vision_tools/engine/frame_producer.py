"""
FrameProducer — Extracts frames from video sources with metadata.

Decoupled from the pipeline — just reads frames, computes scene-change
scores, builds FrameMetadata, and pushes to an asyncio Queue.
"""
import asyncio
import logging
import os
from typing import Optional

import cv2
import numpy as np

from ..utils.types import FrameContext
from ..utils.schemas import FrameMetadata
from ..utils.image_utils import color_histogram

logger = logging.getLogger(__name__)


class FrameProducer:
    """
    Async frame producer that reads video frames and emits them
    with computed metadata (frame index, timestamp, scene-change score).
    
    Usage:
        producer = FrameProducer(video_path)
        queue = asyncio.Queue()
        await producer.produce(queue)
        # Frames arrive as (np.ndarray, FrameMetadata) tuples; None signals end.
    """

    def __init__(self, video_path: str, camera_id: str = "default"):
        self.video_path = self._resolve_video_source(video_path)
        self.camera_id = camera_id
        self.video_fps: Optional[float] = None
        self.total_frames: int = 0
        self._last_frame: Optional[np.ndarray] = None
        self._frame_idx: int = 0

    async def produce(self, queue: asyncio.Queue, 
                      skip_frames: int = 0) -> None:
        """
        Read frames from video and push (frame, metadata) tuples to queue.
        Sends None when done.
        
        Args:
            queue: Async queue to push frames to
            skip_frames: Skip every N frames (0 = no skip)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {self.video_path}")
            await queue.put(None)
            return

        try:
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(
                f"FrameProducer: {self.video_path} — "
                f"{self.total_frames} frames @ {self.video_fps:.1f} fps"
            )

            while True:
                result = await asyncio.to_thread(self._read_frame, cap)
                if result is None:
                    break

                frame, metadata = result
                
                # Optional frame skipping
                if skip_frames > 0 and metadata.frame_idx % (skip_frames + 1) != 0:
                    continue

                await queue.put((frame, metadata))

        except Exception as e:
            logger.error(f"FrameProducer error: {e}")
        finally:
            cap.release()
            await queue.put(None)  # Signal end of stream

    def _read_frame(self, cap: cv2.VideoCapture):
        """Read a single frame and compute metadata (runs in thread)."""
        ret, frame = cap.read()
        if not ret:
            return None

        # Compute scene-change score
        if self._last_frame is not None:
            scene_change_score = self._hist_distance(self._last_frame, frame)
        else:
            scene_change_score = 1.0

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        metadata = FrameMetadata(
            frame_idx=self._frame_idx,
            timestamp=timestamp,
            scene_change_score=scene_change_score,
            camera_id=self.camera_id,
        )

        self._last_frame = frame
        self._frame_idx += 1

        return frame, metadata

    @staticmethod
    def _hist_distance(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Bhattacharyya distance between color histograms."""
        hist1 = color_histogram(frame1)
        hist2 = color_histogram(frame2)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    @staticmethod
    def _resolve_video_source(video_path: str) -> str:
        """Resolve video path (local file or URL)."""
        if os.path.exists(video_path):
            return os.path.abspath(video_path)
        if "youtube.com" in video_path or "youtu.be" in video_path:
            raise ValueError("YouTube links are not supported.")
        return video_path
