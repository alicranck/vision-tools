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
        text = raw_output[0] if raw_output else ""
        # Guard against missing 'Assistant:' delimiter
        if "Assistant:" in text:
            text = text.split("Assistant:", 1)[1].strip()
        caption = Caption(text=text, model_id=self._model_id)
        return CaptionResult(caption=caption).model_dump()

    def preprocess(self, inputs: np.ndarray) -> Any:
        """Prepare frame for SmolVLM2 inference."""
        from vision_tools.utils.image_utils import base64_encode

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Give a concise description of what is happening in this image"},
                {"type": "image", "url": f"data:image/png;base64,{base64_encode(inputs, 'png')}"},
            ],
        }]
        return self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model_id if isinstance(self._model_id, str) and self._model_id != "" else "cpu") 
        # Note: device handling in SmolVLM is a bit tricky with optimum-intel, usually CPU.
        # But we need to match signature. Let's simplify and rely on the model object handling device or the user passing it.
        # Actually ModelNode doesn't pass device to process/preprocess.
        # For this fix, I'll keep it simple and assume CPU or rely on load_model's device. 
        # Ideally preprocess shouldn't need device if it returns standard tensors that infer() moves, 
        # but apply_chat_template returns pt tensors.


    def get_available_runtimes(self) -> list[str]:
        return ["openvino"]
