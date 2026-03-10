from __future__ import annotations

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.graph_types import Embedding, Image
from vision_tools.nodes.model.model_node import ModelNode


class EmbedderConfig(BaseModel):
    model: str = Field("siglip2", description="Model backend: 'siglip2', 'clip'")
    runtime: str = Field("auto", description="Runtime: 'auto', 'pytorch', 'openvino'")


class Embedder(ModelNode):
    InputPorts = {"image": "Image"}
    OutputPorts = {"embedding": "Embedding"}

    def __init__(self, node_id: str = "embedder", config: dict | None = None, **kwargs) -> None:
        config = config or {}
        model_name = config.get("model", "siglip2")
        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(task="embedding", model=model_name)
        super().__init__(node_id=node_id, config=config, **kwargs)

    def preprocess(self, inputs, context):
        return inputs["image"].data

    def normalize_outputs(self, outputs):
        if isinstance(outputs, Embedding):
            return {"embedding": outputs}
        if isinstance(outputs, dict) and "embedding" in outputs and isinstance(outputs["embedding"], Embedding):
            return outputs
        if not isinstance(outputs, dict):
            raise TypeError(f"{self.node_id}: backend output must be a dict.")
        embedding = outputs.get("embedding", outputs)
        if hasattr(embedding, "model_dump"):
            embedding = embedding.model_dump()
        return {"embedding": Embedding.model_validate(embedding)}

    def encode_text(self, text: str) -> list[float]:
        if not hasattr(self.backend, "encode_text"):
            raise NotImplementedError(
                f"Backend {self.backend.__class__.__name__} "
                "does not support text encoding."
            )
        return self.backend.encode_text(self.model, text)

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return EmbedderConfig
