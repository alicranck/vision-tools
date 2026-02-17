from pathlib import Path
import asyncio
import logging
from collections import defaultdict
from typing import Optional, Type

from trackers import SORTTracker
import supervision as sv
from ultralytics import YOLOE  # type: ignore
from ultralytics.engine.results import Boxes  # type: ignore
import numpy as np
from pydantic import BaseModel

from .base_tool import BaseVisionTool, ToolKey
from ...utils.tracking import BoxKalmanFilter
from ...utils.types import ImageHandle, List, Any, Dict
from ...utils.schemas import BoundingBox, DetectionResult


logger = logging.getLogger(__name__)


DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


class OpenVocabularyDetector(BaseVisionTool):
    """
    Detection tool using an open-vocabulary YOLO model.
    Used for unconstrained zero-shot object detection based on a custom vocabulary.
    Supports fine-tuning via Ultralytics training API.
    """
    # Typed I/O contract for pipeline validation
    OutputSchema = DetectionResult
    InputSchema = None  # No upstream tool dependency

    def __init__(self, model_id, config, device = 'cpu'):
        self.imgsz: int
        self.conf_threshold: float
        self.vocabulary: List[str] | None
        self.tracker: SORTTracker
        self.kalman_filters: Dict[str, BoxKalmanFilter] = {}
        super().__init__(model_id, config, device)

    def _configure(self, config: dict):
        self.imgsz = config.get('imgsz', DEFAULT_IMAGE_SIZE)
        self.conf_threshold = config.get('conf_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        self.vocabulary = config.get('vocabulary', None)
        self.prompt_free = config.get('prompt_free', False)
        return

    def _load_model(self):
        """
        Loads the YOLOE model and initializes the SORT tracker.
        Automatically selects optimal backend (CUDA or OpenVINO).
        
        Returns:
            The model ready for inference.
        """
        if not self.prompt_free and self.vocabulary is None:
            raise ValueError("OpenVocabularyDetector requires a 'vocabulary' list in the config, unless prompt_free=True.")
        
        model_name = self.model_id # Initialize with model_id
        if self.prompt_free:
             model_name = "yoloe-11s-seg-pf.pt" 
        
        resolved_path = self._resolve_model_path(model_name)
        
        model = YOLOE(resolved_path)
        logger.debug(f"Loaded model: {resolved_path}")

        if not self.prompt_free:
            pos_embeddings = model.get_text_pe(self.vocabulary)
            model.set_classes(self.vocabulary, pos_embeddings)
        
        # Select backend based on hardware
        if self.device == "cuda":
            # GPU: Keep native YOLO (already optimized for CUDA)
            compiled_model = model
            logger.info(f"{self.tool_name}: Using native CUDA backend")
        else:
            # CPU: Export to OpenVINO for Intel optimization
            compiled_model = self.compile_ov_model(model, imgsz=self.imgsz)
            logger.info(f"{self.tool_name}: Using OpenVINO backend")

        self.tracker = SORTTracker(lost_track_buffer=5, frame_rate=10, 
                                    minimum_consecutive_frames=2,
                                    minimum_iou_threshold=0.2)
        self.tracking_history = defaultdict(list)

        return compiled_model

    def set_vocabulary(self, classes: list):
        if self.model:
            self.model.set_classes(classes)
            logger.info(f"DetectionTool vocabulary set to: {classes}")

    def inference(self, model_inputs: np.ndarray) -> Any:
        """Runs YOLO inference."""
        results = self.model.predict(model_inputs,
                                    conf=self.conf_threshold,
                                    imgsz=self.imgsz)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = self.tracker.update(detections)

        self.extrapolated_frames = 0

        for i, track_id in enumerate(detections.tracker_id):
            if track_id is None or track_id == -1:
                continue
            # Update Kalman Filter
            if track_id not in self.kalman_filters:
                self.kalman_filters[track_id] = BoxKalmanFilter(detections.xyxy[i], 
                                                                detections.class_id[i], 
                                                                detections.confidence[i])
            else:
                self.kalman_filters[track_id].update(detections.xyxy[i])
                
        # Remove finished tracks
        finished_tracks = set(self.kalman_filters.keys()) - set(detections.tracker_id)
        for ft_id in finished_tracks:
            del self.kalman_filters[ft_id]

        return {"tracks": self.kalman_filters, "class_names": results[0].names}

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        """Parses YOLO results into typed BoundingBox schemas."""
        class_names = raw_output["class_names"]
        boxes = [
            BoundingBox(
                xyxy=list(map(float, kf.xyxy)),
                class_id=int(kf.class_idx),
                confidence=float(kf.conf),
                tracker_id=int(tid),
                class_name=class_names.get(int(kf.class_idx)),
            )
            for tid, kf in raw_output["tracks"].items()
        ]
        result = DetectionResult(boxes=boxes, class_names=class_names)
        return result.model_dump()
    
    def extrapolate_last(self, frame_handle: ImageHandle) -> Any:
        for track_id, kalman_filter in self.kalman_filters.items():
            new_xyxy = kalman_filter.predict()
            kalman_filter.update(new_xyxy)

        results = self.postprocess({"tracks": self.kalman_filters,
                                    "class_names": self.last_result["class_names"]},
                                    None)

        return results

    # ------------------------------------------------------------------
    # Training support (Ultralytics YOLO fine-tuning)
    # ------------------------------------------------------------------

    async def _train_impl(self, dataset_ref: str, config: dict) -> None:
        """
        Fine-tune the YOLO model on a user-provided dataset.
        
        Args:
            dataset_ref: Path to a YOLO-format dataset YAML file.
            config: Training hyperparameters:
                - epochs (int): Number of training epochs (default: 50)
                - batch_size (int): Batch size (default: 16)
                - imgsz (int): Training image size (default: 640)
                - lr0 (float): Initial learning rate (default: 0.01)
                - save_dir (str): Directory to save checkpoints
        """
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 16)
        imgsz = config.get('imgsz', self.imgsz)
        lr0 = config.get('lr0', 0.01)
        save_dir = config.get('save_dir', None)

        train_kwargs = {
            'data': dataset_ref,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'lr0': lr0,
            'device': self.device,
            'verbose': True,
        }
        if save_dir:
            train_kwargs['project'] = save_dir

        logger.info(f"{self.tool_name}: Starting YOLO training with config: {train_kwargs}")
        
        # Run training in a thread to avoid blocking the async loop
        results = await asyncio.to_thread(
            self.model.train, **train_kwargs
        )
        
        logger.info(f"{self.tool_name}: Training complete. Results: {results}")

    # ------------------------------------------------------------------
    # Model download and compilation
    # ------------------------------------------------------------------

    def download_ckpt(self, model_id: str, destination: Path) -> Path:
        """
        Manually downloads the YOLO checkpoint if not found.
        """
        model = YOLOE(destination)
        del model
        return destination

    @staticmethod
    def compile_ov_model(model, imgsz):
        exported_model_path = model.export(format="openvino", simplify=True,
                                nms=True, imgsz=imgsz, batch=1, dynamic=True)
        ov_model = YOLOE(exported_model_path)
        return ov_model
    
    @property
    def output_keys(self) -> list:
        boxes = ToolKey(
            key_name="boxes",
            data_type=Boxes,
            description="Detected bounding boxes",
        )
        return [boxes]
    
    @property
    def processing_input_keys(self) -> list:
        return []

    @property
    def config_keys(self) -> list:
        vocabulary = ToolKey(
            key_name="vocabulary",
            data_type=List[str],
            description="List of classes for open-vocabulary detection (required unless prompt_free=True)",
            required=False
        )
        prompt_free = ToolKey(
            key_name="prompt_free",
            data_type=bool,
            description="Use strict prompt-free model (default: False)",
            required=False
        )
        image_size = ToolKey(
            key_name="image_size",
            data_type=int,
            description="Image size for model input (default: 640)",
        )
        confidence = ToolKey(
            key_name="confidence_threshold",
            data_type=float,
            description="Confidence threshold for detections (default: 0.25)",
        )
        return [vocabulary, prompt_free, image_size, confidence]
