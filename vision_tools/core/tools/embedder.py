from typing import Any, List
import clip
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
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
    @property
    def config_keys(self) -> List[ToolKey]:
        return []


class JinaEmbedder(BaseVisionTool):
    """
    Tool for generating multimodal embeddings using jinaai/jina-embeddings-v4.
    """
    def __init__(self, model_id: str = "jinaai/jina-embeddings-v4", 
                                config: dict = None, device: str = 'cpu'):
        if config is None:
            config = {}
        super().__init__(model_id, config, device)

    def _load_model(self) -> Any:
        model = AutoModel.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            device_map=self.device
        )
        return model

    def preprocess(self, frame: np.ndarray) -> Any:
        pil_image = Image.fromarray(frame)
        return pil_image

    def preprocess_text(self, text: str) -> Any:
        return text

    def inference(self, model_inputs: Any) -> Any:
        embeddings = self.model.encode_image(model_inputs, task="retrieval")
        return embeddings

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        if isinstance(raw_output, torch.Tensor):
             raw_output = raw_output.cpu().numpy()
        
        if isinstance(raw_output, np.ndarray):
             embedding = raw_output.tolist()
             if isinstance(embedding[0], list): # Batch size 1
                 embedding = embedding[0]
        else:
            embedding = raw_output

        return {"embedding": embedding}

    def encode_text(self, text: str) -> List[float]:
        if not self.loaded:
            raise RuntimeError(f"ERROR: {self.tool_name} is not loaded. Call .load_tool() first.")
        
        embeddings = self.model.encode([text], task="retrieval", prompt_name="query")
        return embeddings[0].tolist()

    @property
    def output_keys(self) -> List[ToolKey]:
        return [
            ToolKey("embedding", list, "Jina embedding vector of the image or text")
        ]

    @property
    def processing_input_keys(self) -> List[ToolKey]:
        return []

    @property
    def config_keys(self) -> List[ToolKey]:
        return []


class SigLIP2Embedder(BaseVisionTool):
    """
    Tool for generating multimodal embeddings using google siglip2.
    """
    def __init__(self, model_id: str = "google/siglip2-base-patch16-384",
                                     config: dict = None, device: str = 'cpu'):
        if config is None:
            config = {}
        super().__init__(model_id, config, device)

    def _load_model(self) -> Any:
        model = AutoModel.from_pretrained(self.model_id).eval()
        model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return model

    def preprocess(self, frame: np.ndarray) -> Any:
        pil_image = Image.fromarray(frame)
        inputs = self.processor(images=[pil_image], return_tensors="pt")
        return inputs.to(self.device)

    def preprocess_text(self, text: str) -> Any:
        tokens = self.tokenizer([text], padding="max_length",
                    max_length=64, return_tensors="pt")
        return tokens.to(self.device)

    def inference(self, model_inputs: Any) -> Any:
        embeddings = self.model.get_image_features(**model_inputs)
        return embeddings

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        embedding = raw_output.cpu().numpy().tolist()
        return {"embedding": embedding}

    def encode_text(self, text: str) -> List[float]:
        if not self.loaded:
            raise RuntimeError(f"ERROR: {self.tool_name} is not loaded. Call .load_tool() first.")
        
        text_input = self.preprocess_text(text)
        with torch.no_grad():
             raw_output = self.model.get_text_features(**text_input)
        
        return raw_output.cpu().numpy().tolist()

    @property
    def output_keys(self) -> List[ToolKey]:
        return [
            ToolKey("embedding", list, "SigLIP2 embedding vector of the image or text")
        ]

    @property
    def processing_input_keys(self) -> List[ToolKey]:
        return []

    @property
    def config_keys(self) -> List[ToolKey]:
        return []
