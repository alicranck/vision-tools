import asyncio
import cv2
import os
import traceback
import logging
from ..core.tools.pipeline import VisionPipeline
from ..utils.image_utils import color_histogram
from ..utils.types import FrameContext


logger = logging.getLogger(__name__)


DELAY_SECONDS_DEFAULT = 3.0
MAX_QUEUE_SIZE = 300

class VideoInferenceEngine:
    """
    Orchestrates the video processing pipeline.
    Handles video reading, pipeline execution, and frame serving.
    Uses a producer-consumer pattern to ensure smooth streaming.
    """
    def __init__(self, tool_pipeline: VisionPipeline, video_path: str):
        """
        Initializes the VideoInferenceEngine.

        Args:
            tool_pipeline (VisionPipeline): The vision pipeline to process frames.
            video_path (str): Path to the video file or URL.
        """
        self.video_path = self._resolve_video_source(video_path)
        self.tool_pipeline = tool_pipeline
        self.video_fps = None
        self.last_frame_idx = -1
        self.last_frame = None

    async def run_inference(self, on_data=None, 
                             buffer_delay: float = DELAY_SECONDS_DEFAULT,
                              max_queue_size: int = MAX_QUEUE_SIZE,
                              realtime: bool = True):
        """
        Starts the inference process and yields processed frames.
        
        Args:
            on_data (callable, optional): Async callback for sending metadata to the client.
            buffer_delay (float): Time in seconds to buffer before starting the stream.
            max_queue_size (int): Maximum number of items in the producer-consumer queue.
            
        Yields:
            bytes: MJPEG frame chunks.
        """
        queue = asyncio.Queue(maxsize=max_queue_size)
        producer_task = asyncio.create_task(self._inference_producer(queue))

        if realtime:
            logger.info(f"Buffering for {buffer_delay} seconds...")
            await asyncio.sleep(buffer_delay)

        try:
            while True:
                if producer_task.done() and queue.empty():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if item is None:
                        break

                    frame_bytes, data = item
                    if on_data:
                        await on_data(data)

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                    if realtime and self.video_fps and self.video_fps > 0:
                        await asyncio.sleep(1.0 / self.video_fps)

                except asyncio.TimeoutError:
                    continue
        
        except Exception as e:
            logger.error(f"Streaming Error: {e}")
            logger.error(traceback.format_exc())
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    async def _inference_producer(self, queue: asyncio.Queue):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video stream: {self.video_path}")
            await queue.put(None)
            return
        try:
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                result = await asyncio.to_thread(self._process_next_frame, cap)                
                if result is None:
                    break

                await queue.put(result)

        except Exception as e:
            logger.error(f"Producer Error: {e}")
            logger.error(traceback.format_exc())
        finally:
            cap.release()
            await queue.put(None)  # Signal end of stream

    def _process_next_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None

        scene_change_score = 0.0
        if self.last_frame is not None:
            scene_change_score = self.hist_distance(self.last_frame, frame)
        else:
            scene_change_score = 1.0

        context = FrameContext(frame_idx=self.last_frame_idx + 1,
                                scene_change_score=scene_change_score,
                                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        
        processed_frame, data = self.tool_pipeline.run_pipeline(frame, context=context)
        
        self.last_frame_idx += 1
        self.last_frame = frame

        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        
        return frame_bytes, data
            
    @staticmethod
    def hist_distance(frame1, frame2) -> float:
        hist1 = color_histogram(frame1)
        hist2 = color_histogram(frame2)
        dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return dist

    def _resolve_video_source(self, video_path: str) -> str:
        """
        Resolves the video source from a path or URL.
        Handles local files, YouTube links, and direct URLs.
        """
        if os.path.exists(video_path):
            return os.path.abspath(video_path)
        
        if "youtube.com" in video_path or "youtu.be" in video_path:
            raise ValueError("YouTube links are not supported.")

        return video_path

