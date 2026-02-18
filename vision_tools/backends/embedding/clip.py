"""
CLIPBackend — OpenAI CLIP model loading, inference, postprocessing.

Ported from ``CLIPEmbedder``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext
from vision_tools.core.schemas import Embedding, EmbeddingResult

logger = logging.getLogger(__name__)


@BackendRegistry.register(task="embedding", model="clip")
class CLIPBackend:
    """Backend for OpenAI CLIP embedding models."""

    def __init__(self) -> None:
        self._preprocess_fn = None
        self._model_id = ""
        self._device = "cpu"

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        """Load CLIP model."""
        import clip

        self._model_id = model_path
        actual_device = "cuda" if device == "cuda" else "cpu"
        self._device = actual_device

        model, preprocess = clip.load(model_path, device=actual_device)
        self._preprocess_fn = preprocess
        return model

    def infer(self, model: Any, inputs: Any) -> Any:
        """Run CLIP image encoding."""
        return model.encode_image(inputs)

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        """Convert CLIP features to EmbeddingResult dict."""
        features = raw_output / raw_output.norm(dim=-1, keepdim=True)
        vector = features.cpu().numpy()[0].tolist()
        emb = Embedding(
            vector=vector,
            model_id=self._model_id,
            dimension=len(vector),
        )
        return EmbeddingResult(embedding=emb).model_dump()

    def encode_text(self, model: Any, text: str) -> list[float]:
        """Encode text using CLIP."""
        import clip
        import torch

        text_input = clip.tokenize(text).to(self._device)
        with torch.no_grad():
            raw = model.encode_text(text_input)
        result = self.postprocess(raw, NodeContext())
        return result["embedding"]["vector"]

    def preprocess_image(self, frame: np.ndarray) -> Any:
        """Preprocess image for CLIP."""
        from PIL import Image
        pil_image = Image.fromarray(frame)
        return self._preprocess_fn(pil_image).unsqueeze(0).to(self._device)

    def get_available_runtimes(self) -> list[str]:
        return ["pytorch"]
