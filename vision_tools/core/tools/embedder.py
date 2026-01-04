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

    def preprocess_text(self, text: str) -> Any:
        text_input = clip.tokenize(text).to(self.device)
        return text_input

    def inference(self, model_inputs: Any) -> Any:
        features = self.model.encode_image(model_inputs)
        return features

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        features = raw_output / raw_output.norm(dim=-1, keepdim=True)
        embedding = features.cpu().numpy()[0].tolist()
        return {"embedding": embedding}

    def encode_text(self, text: str) -> List[float]:
        """
        Simple method for encoding text directly without pipeline context.
        Useful for search queries.
        """
        if not self.loaded:
            raise RuntimeError(f"ERROR: {self.tool_name} is not loaded. Call .load_tool() first.")
        
        text_input = self.preprocess_text(text)
        with torch.no_grad():
            raw_output = self.model.encode_text(text_input)
        result = self.postprocess(raw_output, None)
        return result["embedding"]

    @property
    def output_keys(self) -> List[ToolKey]:
        return [
            ToolKey("embedding", list, "CLIP embedding vector of the image or text")
        ]

    @property
    def processing_input_keys(self) -> List[ToolKey]:
        return []

    @property
    def config_keys(self) -> List[ToolKey]:
        return []
