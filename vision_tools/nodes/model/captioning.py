from __future__ import annotations

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.graph_types import Caption
from vision_tools.nodes.model.model_node import ModelNode


class CaptionerConfig(BaseModel):
    model: str = Field("smolvlm", description="Model backend: 'smolvlm', 'llamacpp'")
    runtime: str = Field("auto", description="Runtime: 'auto', 'openvino', 'cpu'")
    imgsz: int = Field(512, description="Input image size for preprocessing")
    max_tokens: int = Field(64, description="Max tokens to generate")


class Captioner(ModelNode):
    InputPorts = {"image": "Image"}
    OutputPorts = {"caption": "Caption"}

    def __init__(self, node_id: str = "captioner", config: dict | None = None, **kwargs) -> None:
        config = config or {}
        model_name = config.get("model", "smolvlm")
        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(task="captioning", model=model_name)
        super().__init__(node_id=node_id, config=config, **kwargs)

    def preprocess(self, inputs, context):
        return inputs["image"].data

    def normalize_outputs(self, outputs):
        if not isinstance(outputs, dict):
            raise TypeError(f"{self.node_id}: backend output must be a dict.")
        if "caption" not in outputs:
            raise TypeError(f"{self.node_id}: backend output must define 'caption'.")
        caption = outputs["caption"]
        if hasattr(caption, "model_dump"):
            caption = caption.model_dump()
        return {"caption": Caption.model_validate(caption)}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return CaptionerConfig
