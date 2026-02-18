"""
SigLIP2Backend — SigLIP2 model loading, inference, postprocessing.

Ported from ``SigLIP2Embedder`` and ``OVSigLIP2Embedder``.
Supports PyTorch (CUDA/CPU) and OpenVINO runtimes in a single class.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext
from vision_tools.core.schemas import Embedding, EmbeddingResult

logger = logging.getLogger(__name__)


@BackendRegistry.register(task="embedding", model="siglip2")
class SigLIP2Backend:
    """Backend for Google SigLIP2 embedding models.

    Supports:
    - PyTorch with CUDA
    - PyTorch CPU
    - OpenVINO (Intel CPU optimization)

    Runtime is auto-selected based on device parameter.
    """

    def __init__(self) -> None:
        self._processor = None
        self._tokenizer = None
        self._use_openvino = False
        self._model_id = ""
        self._device = "cpu"

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        """Load SigLIP2 model with auto runtime selection."""
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self._model_id = model_path
        self._device = device

        if device == "cuda":
            import torch
            model = AutoModel.from_pretrained(model_path).eval().to("cuda")
            logger.info("SigLIP2Backend: using CUDA")
        elif device in ("openvino", "ov") or (device == "auto" and self._openvino_available()):
            model = self._load_openvino(model_path)
            self._use_openvino = True
            logger.info("SigLIP2Backend: using OpenVINO")
        else:
            model = AutoModel.from_pretrained(model_path).eval()
            logger.info("SigLIP2Backend: using PyTorch CPU")

        self._processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model

    def infer(self, model: Any, inputs: Any) -> Any:
        """Run SigLIP2 image embedding inference."""
        if self._use_openvino:
            results = model(**inputs)
            return results.image_embeds
        else:
            import torch
            with torch.no_grad():
                return model.get_image_features(**inputs)

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        """Convert raw embeddings to EmbeddingResult dict."""
        import torch
        if isinstance(raw_output, torch.Tensor):
            vector = raw_output.cpu().numpy().squeeze().tolist()
        else:
            vector = raw_output.squeeze().tolist()

        emb = Embedding(
            vector=vector,
            model_id=self._model_id,
            dimension=len(vector),
        )
        return EmbeddingResult(embedding=emb).model_dump()

    def encode_text(self, model: Any, text: str) -> list[float]:
        """Encode text into an embedding vector.

        Called by ``Embedder.encode_text()``.
        """
        import torch
        from PIL import Image

        if self._use_openvino:
            dummy_image = Image.new("RGB", (384, 384))
            inputs = self._processor(
                images=[dummy_image], text=[text],
                max_length=64, padding="max_length",
                return_tensors="pt",
            )
            results = model(**inputs)
            embedding = results.text_embeds
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            return embedding.squeeze().tolist()
        else:
            tokens = self._tokenizer(
                [text], padding="max_length",
                max_length=64, return_tensors="pt",
            )
            if self._device == "cuda":
                tokens = tokens.to("cuda")
            with torch.no_grad():
                raw = model.get_text_features(**tokens)
            return raw.cpu().numpy().squeeze().tolist()

    def get_available_runtimes(self) -> list[str]:
        return ["pytorch", "openvino"]

    def preprocess_image(self, frame: np.ndarray) -> Any:
        """Preprocess a frame for SigLIP2 inference.

        Called by the task node's preprocess() if overridden.
        """
        from PIL import Image
        pil_image = Image.fromarray(frame)
        inputs = self._processor(
            images=[pil_image], text=["dummy"],
            return_tensors="pt",
        )
        if self._device == "cuda":
            return inputs.to("cuda")
        return inputs

    @staticmethod
    def _openvino_available() -> bool:
        try:
            from optimum.intel.openvino import OVModelForZeroShotImageClassification
            return True
        except ImportError:
            return False

    def _load_openvino(self, model_path: str) -> Any:
        from optimum.intel.openvino import OVModelForZeroShotImageClassification
        from optimum.intel.openvino.configuration import OVWeightQuantizationConfig

        ov_path = Path(model_path) / "ov"
        if ov_path.exists():
            return OVModelForZeroShotImageClassification.from_pretrained(
                ov_path, device="cpu"
            )
        else:
            quant_config = OVWeightQuantizationConfig()
            model = OVModelForZeroShotImageClassification.from_pretrained(
                model_path, export=True,
                quantization_config=quant_config, device="cpu",
            )
            model.save_pretrained(ov_path)
            return model
