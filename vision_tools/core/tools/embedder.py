from typing import Any, List
import clip
import torch
from PIL import Image
import numpy as np

from .base_tool import BaseVisionTool, ToolKey

class CLIPEmbedder(BaseVisionTool):
    """
    Tool for generating image embeddings using OpenAI's CLIP model.
    """
    def __init__(self, model_id: str, config: dict, device: str = 'cpu'):
        super().__init__(model_id, config, device)

    def _load_model(self) -> Any:
        model, preprocess = clip.load(self.model_id, device=self.device)
        self.preprocess_fn = preprocess
        return model

    def preprocess(self, frame: np.ndarray) -> Any:
        pil_image = Image.fromarray(frame)
        image_input = self.preprocess_fn(pil_image).unsqueeze(0).to(self.device)
        return image_input

    def inference(self, model_inputs: Any) -> Any:
        image_features = self.model.encode_image(model_inputs)
        return image_features

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        image_features = raw_output / raw_output.norm(dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy()[0].tolist()
        return {"embedding": embedding}

    @property
    def output_keys(self) -> List[ToolKey]:
        return [
            ToolKey("embedding", list, "CLIP embedding vector of the image")
        ]

    @property
    def processing_input_keys(self) -> List[ToolKey]:
        return []

    @property
    def config_keys(self) -> List[ToolKey]:
        return []
