"""
SmolVLMBackend — SmolVLM2 captioning model.

Ported from ``Captioner`` (the SmolVLM2 variant).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.node import NodeContext
from vision_tools.core.schemas import Caption, CaptionResult

logger = logging.getLogger(__name__)


@BackendRegistry.register(task="captioning", model="smolvlm")
class SmolVLMBackend:
    """Backend for SmolVLM2 image captioning (OpenVINO)."""

    def __init__(self) -> None:
        self._processor = None
        self._tokenizer = None
        self._model_id = ""

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        """Load SmolVLM2 model via optimum-intel."""
        from transformers import AutoProcessor, AutoTokenizer
        from optimum.intel.openvino.modeling_visual_language import OVModelForVisualCausalLM

        self._model_id = model_path
        model = OVModelForVisualCausalLM.from_pretrained(model_path)
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model

    def infer(self, model: Any, inputs: Any) -> Any:
        """Generate caption tokens."""
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        """Extract caption text from generated output."""
        assistant_response = raw_output[0].split("Assistant:")[1].strip()
        caption = Caption(text=assistant_response, model_id=self._model_id)
        return CaptionResult(caption=caption).model_dump()

    def preprocess_frame(self, frame: np.ndarray, device: str = "cpu") -> Any:
        """Prepare frame for SmolVLM2 inference."""
        from vision_tools.utils.image_utils import base64_encode

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Give a concise description of what is happening in this image"},
                {"type": "image", "url": f"data:image/png;base64,{base64_encode(frame, 'png')}"},
            ],
        }]
        return self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

    def get_available_runtimes(self) -> list[str]:
        return ["openvino"]
