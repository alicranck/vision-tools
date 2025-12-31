import os
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Keypoints
from .base_tool import BaseVisionTool, ToolKey
from ...utils.types import Any, List
from ...utils.locations import APP_DIR

DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class PoseEstimator(BaseVisionTool):
    """
    Pose estimation tool using Ultralytics YOLO-Pose models.
    """
    def __init__(self, model_id, config, device='cpu'):
        self.imgsz: int
        self.conf_threshold: float
        super().__init__(model_id, config, device)

    def _configure(self, config: dict):
        self.imgsz = config.get('imgsz', DEFAULT_IMAGE_SIZE)
        self.conf_threshold = config.get('conf_threshold', DEFAULT_CONFIDENCE_THRESHOLD)

    def _load_model(self):
        """
        Loads the YOLO-Pose model.
        """
        if not os.path.isfile(self.model_id):
            model_path = APP_DIR / "models" / self.model_id
            if not model_path.exists():
                 model = YOLO(self.model_id)
            else:
                model = YOLO(str(model_path))
        else:
            model = YOLO(self.model_id)
            
        return model

    def inference(self, model_inputs: np.ndarray) -> Any:
        """Runs Pose inference."""
        results = self.model.predict(model_inputs,
                                    conf=self.conf_threshold,
                                    imgsz=self.imgsz,
                                    verbose=False)
        return results[0]

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        """Parses Pose results."""
        # raw_output is a Results object
        keypoints = raw_output.keypoints
        
        output_kpts = []
        if keypoints is not None:
            for i, kpt in enumerate(keypoints):
                # kpt.xy is (1, 17, 2) -> (17, 2)
                xy = kpt.xy[0].cpu().numpy().tolist()
                conf = kpt.conf[0].cpu().numpy().tolist() if kpt.conf is not None else [1.0]*17
                
                output_kpts.append({
                    "id": i,
                    "keypoints": list(zip(xy, conf)), # List of ([x, y], conf)
                })

        return {
            "poses": output_kpts
        }
    
    def extrapolate_last(self, frame_handle: Any) -> Any:
        results = self.postprocess(self.last_result, None)
        return results

    @property
    def output_keys(self) -> list:
        poses = ToolKey(
            key_name="poses",
            data_type=List[dict],
            description="Detected poses with keypoints and bounding boxes",
        )
        return [poses]
    
    @property
    def processing_input_keys(self) -> list:
        return []

    @property
    def config_keys(self) -> list:
        image_size = ToolKey(
            key_name="image_size",
            data_type=int,
            description="Image size for model input (default: 640)",
        )
        confidence = ToolKey(
            key_name="confidence_threshold",
            data_type=float,
            description="Confidence threshold for detections (default: 0.5)",
        )
        return [image_size, confidence]
