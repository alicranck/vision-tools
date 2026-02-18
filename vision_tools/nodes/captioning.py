"""
Captioner — Task node for image captioning.

Supports multiple captioning models (SmolVLM, LlamaCpp) via backend config.

Usage::

    {"node_type": "captioner",
     "config": {"model": "smolvlm"}}
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.schemas import CaptionResult
from vision_tools.nodes.model_node import ModelNode

logger = logging.getLogger(__name__)


class CaptionerConfig(BaseModel):
    """Config schema for Captioner."""
    model: str = Field("smolvlm", description="Model backend: 'smolvlm', 'llamacpp'")
    runtime: str = Field("auto", description="Runtime: 'auto', 'openvino', 'cpu'")
    imgsz: int = Field(512, description="Input image size for preprocessing")
    max_tokens: int = Field(64, description="Max tokens to generate")


@NodeRegistry.register("captioner", category="captioning")
class Captioner(ModelNode):
    """Generate text captions from images.

    The model backend (SmolVLM, LlamaCpp, etc.) is resolved from config.
    """

    OutputSchema = CaptionResult
    InputSchema = None

    def __init__(self, node_id: str = "captioner", config: dict | None = None, **kwargs) -> None:
        config = config or {}
        model_name = config.get("model", "smolvlm")
        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(task="captioning", model=model_name)
        super().__init__(node_id=node_id, config=config, **kwargs)

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return CaptionerConfig
