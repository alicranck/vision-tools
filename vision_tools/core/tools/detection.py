import os
from pathlib import Path
import logging
from collections import defaultdict
from trackers import SORTTracker
import supervision as sv
from ultralytics import YOLOE  # type: ignore
from ultralytics.engine.results import Boxes  # type: ignore
import cv2
import numpy as np

from .base_tool import BaseVisionTool, ToolKey
from ...utils.image_utils import load_image_pil
from ...utils.tracking import BoxKalmanFilter
from ...utils.types import ImageHandle, List, NumpyMask, Any
from ...utils.locations import APP_DIR


logger = logging.getLogger(__name__)


DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


class OpenVocabularyDetector(BaseVisionTool):
    """
    Detection tool using an open-vocabulary YOLO model.
    Used for unconstrained zero-shot object detection based on a custom vocabulary.
    """
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
        
        Returns:
            The compiled ONNX/OpenVINO model ready for inference.
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
            
        onnx_model = self.compile_ov_model(model, imgsz=self.imgsz)

        self.tracker = SORTTracker(lost_track_buffer=5, frame_rate=10, 
                                    minimum_consecutive_frames=2,
                                    minimum_iou_threshold=0.2)
        self.tracking_history = defaultdict(list)

        return onnx_model

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
        """Parses YOLO results and updates the data dict."""
        output = {
            "boxes": [{"xyxy": kf.xyxy, "cls": kf.class_idx, "conf": kf.conf, "id": tid} 
                                                for tid, kf in raw_output["tracks"].items()],
            "class_names": raw_output["class_names"]
        }
        return output
    
    def extrapolate_last(self, frame_handle: ImageHandle) -> Any:
        for track_id, kalman_filter in self.kalman_filters.items():
            new_xyxy = kalman_filter.predict()
            kalman_filter.update(new_xyxy)

        results = self.postprocess({"tracks": self.kalman_filters,
                                    "class_names": self.last_result["class_names"]},
                                    None)

        return results

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
