from typing import Any, List
import clip
import torch
import logging
from pathlib import Path
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from optimum.intel.openvino import OVModelForZeroShotImageClassification
from optimum.intel.openvino.configuration import OVWeightQuantizationConfig
from huggingface_hub import snapshot_download
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


class SigLIP2Embedder(BaseVisionTool):
    """
    Tool for generating multimodal embeddings using google siglip2.
    """
    def __init__(self, model_id: str = "google/siglip2-base-patch16-384",
                        config: dict = None, device: str = 'cpu'):
        if config is None:
            config = {}
        self.dummy_text = ["dummy"]
        super().__init__(model_id, config, device)

    def _load_model(self) -> Any:
        
        model_path = self._resolve_model_path(self.model_id)

        model = AutoModel.from_pretrained(model_path).eval()
        model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model

    def download_ckpt(self, model_id: str, destination: str) -> str:
        local_dir = snapshot_download(repo_id=model_id, local_dir=destination)
        return local_dir

    def preprocess(self, frame: np.ndarray) -> Any:
        pil_image = Image.fromarray(frame)
        inputs = self.processor(images=[pil_image], text=self.dummy_text,
                                                         return_tensors="pt")
        return inputs.to(self.device)

    def preprocess_text(self, text: str) -> Any:
        tokens = self.tokenizer([text], padding="max_length",
                    max_length=64, return_tensors="pt")
        return tokens.to(self.device)

    def inference(self, model_inputs: Any) -> Any:
        with torch.no_grad():
            embeddings = self.model.get_image_features(**model_inputs)
        return embeddings

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        embedding = raw_output.cpu().numpy().squeeze().tolist()
        return {"embedding": embedding}

    def encode_text(self, text: str) -> List[float]:
        if not self.loaded:
            raise RuntimeError(f"ERROR: {self.tool_name} is not loaded. Call .load_tool() first.")
        
        text_input = self.preprocess_text(text)
        with torch.no_grad():
             raw_output = self.model.get_text_features(**text_input)
        
        return raw_output.cpu().numpy().squeeze().tolist()

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


class OVSigLIP2Embedder(SigLIP2Embedder):
    """
    OpenVINO optimized version of SigLIP2Embedder.
    """

    def __init__(self, model_id: str = "google/siglip2-base-patch16-384",
                        config: dict = None, device: str = 'cpu'):
        super().__init__(model_id, config, device)
        self.dummy_image = Image.new("RGB", (384, 384))

    def _load_model(self) -> Any:
        
        model_path = self._resolve_model_path(self.model_id)
        
        model = self.export_to_openvino(model_path)
        
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model

    def export_to_openvino(self, model_path: str):
        
        ov_model_path = Path(model_path) / "ov"

        quangization_config = OVWeightQuantizationConfig()

        if ov_model_path.exists():
             model = OVModelForZeroShotImageClassification.from_pretrained(ov_model_path, 
                                                                            quantization_config=quangization_config,
                                                                            device=self.device)
        else:
             model = OVModelForZeroShotImageClassification.from_pretrained(model_path, export=True, quantization_config=quangization_config,
                                                                            device=self.device)
             model.save_pretrained(ov_model_path)
        
        return model

    def inference(self, model_inputs: Any) -> Any:
        results = self.model(**model_inputs)
        return results.image_embeds

    def preprocess_text(self, text: str) -> Any:
        inputs = self.processor(images=[self.dummy_image], text=[text],
                                 max_length=64, padding="max_length", 
                                 return_tensors="pt")
        return inputs.to(self.device)

    def encode_text(self, text: str) -> List[float]:
        if not self.loaded:
            raise RuntimeError(f"ERROR: {self.tool_name} is not loaded. Call .load_tool() first.")
        
        text_input = self.preprocess_text(text)
        results = self.model(**text_input)
        embedding = results.text_embeds
        
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
            
        return embedding.squeeze().tolist()

