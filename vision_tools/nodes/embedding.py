"""
Embedder — Task node for image/text embedding.

Supports multiple embedding models (SigLIP2, CLIP) via backend config.

Usage::

    {"node_type": "embedder",
     "config": {"model": "siglip2"}}
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.schemas import EmbeddingResult
from vision_tools.nodes.model_node import ModelNode

logger = logging.getLogger(__name__)


class EmbedderConfig(BaseModel):
    """Config schema for Embedder — used by LLM agents for discovery."""
    model: str = Field("siglip2", description="Model backend: 'siglip2', 'clip'")
    runtime: str = Field("auto", description="Runtime: 'auto', 'pytorch', 'openvino'")


@NodeRegistry.register("embedder", category="embedding")
class Embedder(ModelNode):
    """Generate embeddings from images or text.

    The model backend (SigLIP2, CLIP, etc.) is resolved from config.
    Backends may expose additional methods like ``encode_text()``
    accessible via ``self.backend``.
    """

    OutputSchema = EmbeddingResult
    InputSchema = None

    def __init__(self, node_id: str = "embedder", config: dict | None = None, **kwargs) -> None:
        config = config or {}
        model_name = config.get("model", "siglip2")
        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(task="embedding", model=model_name)
        super().__init__(node_id=node_id, config=config, **kwargs)

    def encode_text(self, text: str) -> list[float]:
        """Encode text into an embedding vector.

        Delegates to the backend's ``encode_text()`` method.
        Requires the node to be loaded.

        Args:
            text: Input text to encode.

        Returns:
            List of floats representing the text embedding.
        """
        if not hasattr(self.backend, "encode_text"):
            raise NotImplementedError(
                f"Backend {self.backend.__class__.__name__} "
                "does not support text encoding."
            )
        return self.backend.encode_text(self.model, text)

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return EmbedderConfig
