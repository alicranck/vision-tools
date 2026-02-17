"""
DynamicBatcher — Accumulates frames and flushes as BatchPayloads.

Sits between FrameProducer and Pipeline in the async pipeline:
    FrameProducer → DynamicBatcher → Pipeline → output

Flush triggers:
- max_batch_size reached
- max_wait_ms elapsed since first frame in current batch
"""
import asyncio
import logging
import time
import uuid
from typing import Optional

import numpy as np

from ..utils.schemas import BatchPayload, FrameMetadata

logger = logging.getLogger(__name__)


class DynamicBatcher:
    """
    Accumulates frames from a producer queue and emits BatchPayloads
    to a consumer queue.
    
    Args:
        max_batch_size: Maximum frames per batch before flushing
        max_wait_ms: Maximum milliseconds to wait before flushing a partial batch
    """

    def __init__(self, max_batch_size: int = 16, max_wait_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        # Current batch accumulator
        self._frames: list = []
        self._metadatas: list = []
        self._batch_start_time: Optional[float] = None
        self._batches_emitted: int = 0

    async def run(self, input_queue: asyncio.Queue, 
                  output_queue: asyncio.Queue) -> None:
        """
        Consume frames from input_queue, batch them, and push
        BatchPayloads to output_queue.
        
        Sends None to output_queue when all input is consumed.
        """
        logger.info(
            f"DynamicBatcher started: max_batch={self.max_batch_size}, "
            f"max_wait={self.max_wait_ms}ms"
        )

        try:
            while True:
                # Wait for next frame, but with timeout for partial flush
                timeout = self._time_until_flush()
                
                try:
                    item = await asyncio.wait_for(
                        input_queue.get(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Timeout: flush partial batch
                    if self._frames:
                        await self._flush(output_queue)
                    continue

                if item is None:
                    # End of stream: flush remaining and signal done
                    if self._frames:
                        await self._flush(output_queue)
                    break

                frame, metadata = item
                self._accumulate(frame, metadata)

                # Check if we should flush (size trigger)
                if len(self._frames) >= self.max_batch_size:
                    await self._flush(output_queue)

        finally:
            await output_queue.put(None)  # Signal end of batches
            logger.info(f"DynamicBatcher done: {self._batches_emitted} batches emitted")

    def _accumulate(self, frame: np.ndarray, metadata: FrameMetadata):
        """Add a frame to the current batch."""
        if not self._frames:
            self._batch_start_time = time.monotonic()
        self._frames.append(frame)
        self._metadatas.append(metadata)

    async def _flush(self, output_queue: asyncio.Queue):
        """Emit current batch as a BatchPayload."""
        if not self._frames:
            return

        payload = BatchPayload(
            batch_id=f"batch-{self._batches_emitted:04d}-{uuid.uuid4().hex[:8]}",
            frames=list(self._frames),
            frame_metadatas=list(self._metadatas),
        )

        await output_queue.put(payload)
        
        logger.debug(
            f"Flushed batch {payload.batch_id}: {len(payload)} frames "
            f"(waited {self._elapsed_ms():.1f}ms)"
        )

        # Reset accumulator
        self._frames.clear()
        self._metadatas.clear()
        self._batch_start_time = None
        self._batches_emitted += 1

    def _time_until_flush(self) -> Optional[float]:
        """Seconds until max_wait_ms expires, or None if no frames accumulated."""
        if not self._batch_start_time:
            return None  # No timeout — block until frame arrives
        
        elapsed = (time.monotonic() - self._batch_start_time) * 1000  # ms
        remaining = max(0, self.max_wait_ms - elapsed) / 1000  # seconds
        return remaining if remaining > 0 else 0.001  # Avoid zero timeout

    def _elapsed_ms(self) -> float:
        """Milliseconds since batch started."""
        if self._batch_start_time is None:
            return 0.0
        return (time.monotonic() - self._batch_start_time) * 1000
