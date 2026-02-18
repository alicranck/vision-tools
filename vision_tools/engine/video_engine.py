"""
VideoInferenceEngine — Orchestrates video processing with dynamic batching.

Connects: FrameProducer → DynamicBatcher → Pipeline → output

Supports two modes:
1. Streaming (legacy): yields MJPEG frames for real-time display
2. Batch processing: processes entire video and collects all results
"""
import asyncio
import cv2
import logging
import traceback
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import numpy as np

# Support both old VisionPipeline and new Pipeline
try:
    from ..pipeline import Pipeline as NewPipeline
except ImportError:
    NewPipeline = None

from ..core.tools.pipeline import VisionPipeline
from ..utils.schemas import BatchPayload, FrameMetadata, FrameResult
from ..utils.types import FrameContext
from .frame_producer import FrameProducer
from .batcher import DynamicBatcher

logger = logging.getLogger(__name__)


DELAY_SECONDS_DEFAULT = 3.0
MAX_QUEUE_SIZE = 300


class VideoInferenceEngine:
    """
    Orchestrates the video processing pipeline with dynamic batching.
    
    Connects FrameProducer → DynamicBatcher → VisionPipeline, supporting
    both real-time streaming and offline batch processing modes.
    """

    def __init__(self, tool_pipeline: VisionPipeline, video_path: str,
                 max_batch_size: int = 16, max_wait_ms: float = 50.0):
        """
        Args:
            tool_pipeline: The vision pipeline to process frames
            video_path: Path to video file or URL
            max_batch_size: Max frames per batch
            max_wait_ms: Max wait time before flushing partial batch
        """
        self.tool_pipeline = tool_pipeline
        self.producer = FrameProducer(video_path)
        self.batcher = DynamicBatcher(
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )
        self.video_path = self.producer.video_path

    # ------------------------------------------------------------------
    # Mode 1: Real-time streaming (legacy compatible)
    # ------------------------------------------------------------------

    async def run_inference(self, on_data: Optional[Callable] = None,
                           buffer_delay: float = DELAY_SECONDS_DEFAULT,
                           max_queue_size: int = MAX_QUEUE_SIZE,
                           realtime: bool = True) -> AsyncGenerator:
        """
        Stream processed frames as MJPEG chunks.
        Backward-compatible with the original VideoInferenceEngine API.
        
        Args:
            on_data: Async callback for frame metadata
            buffer_delay: Seconds to buffer before streaming
            realtime: If True, pace output to match video FPS
            
        Yields:
            bytes: MJPEG frame chunks
        """
        frame_queue = asyncio.Queue(maxsize=max_queue_size)
        batch_queue = asyncio.Queue(maxsize=max_queue_size // 4)

        # Start producer and batcher as concurrent tasks
        producer_task = asyncio.create_task(
            self.producer.produce(frame_queue)
        )
        batcher_task = asyncio.create_task(
            self.batcher.run(frame_queue, batch_queue)
        )

        if realtime:
            logger.info(f"Buffering for {buffer_delay} seconds...")
            await asyncio.sleep(buffer_delay)

        try:
            while True:
                # Check if everything is done
                if batcher_task.done() and batch_queue.empty():
                    break

                try:
                    batch = await asyncio.wait_for(batch_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if batch is None:
                    break

                # Process the batch through the pipeline
                processed = self.tool_pipeline.process_batch_payload(batch)

                # Yield each frame as MJPEG
                for i, frame in enumerate(processed.frames):
                    metadata = processed.frame_metadatas[i]
                    result = processed.frame_results[i] if i < len(processed.frame_results) else None

                    data = {
                        'metadata': {
                            'frame_idx': metadata.frame_idx,
                            'timestamp': metadata.timestamp,
                            'scene_change_score': metadata.scene_change_score,
                            'tools_run': result.tools_run if result else False,
                        }
                    }
                    if result:
                        data.update(result.results)

                    if on_data:
                        await on_data(data)

                    _, buffer = cv2.imencode(
                        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                    if realtime and self.producer.video_fps and self.producer.video_fps > 0:
                        await asyncio.sleep(1.0 / self.producer.video_fps)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            logger.error(traceback.format_exc())
        finally:
            for task in [producer_task, batcher_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    # ------------------------------------------------------------------
    # Mode 2: Batch processing (new API)
    # ------------------------------------------------------------------

    async def process_video(self, on_batch: Optional[Callable] = None,
                           max_queue_size: int = MAX_QUEUE_SIZE) -> List[FrameResult]:
        """
        Process entire video and return all frame results.
        
        Args:
            on_batch: Optional async callback called with each processed BatchPayload
            max_queue_size: Queue size limit
            
        Returns:
            List of FrameResult for every processed frame
        """
        frame_queue = asyncio.Queue(maxsize=max_queue_size)
        batch_queue = asyncio.Queue(maxsize=max_queue_size // 4)

        all_results: List[FrameResult] = []

        producer_task = asyncio.create_task(
            self.producer.produce(frame_queue)
        )
        batcher_task = asyncio.create_task(
            self.batcher.run(frame_queue, batch_queue)
        )

        try:
            while True:
                if batcher_task.done() and batch_queue.empty():
                    break

                try:
                    batch = await asyncio.wait_for(batch_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if batch is None:
                    break

                processed = self.tool_pipeline.process_batch_payload(batch)
                all_results.extend(processed.frame_results)

                if on_batch:
                    await on_batch(processed)

                logger.info(
                    f"Processed batch {processed.batch_id}: "
                    f"{len(processed)} frames (total: {len(all_results)})"
                )

        finally:
            for task in [producer_task, batcher_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        logger.info(f"Video processing complete: {len(all_results)} frames")
        return all_results

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    @property
    def video_fps(self) -> Optional[float]:
        return self.producer.video_fps

    @staticmethod
    def hist_distance(frame1, frame2) -> float:
        """Legacy: use FrameProducer._hist_distance instead."""
        return FrameProducer._hist_distance(frame1, frame2)
