from __future__ import annotations

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.config import DeviceTarget, InferenceTask, ModelIntent, ModelSize
from vision_tools.core.graph_types import Embedding, Image
from vision_tools.nodes.model.model_node import ModelNode
from vision_tools.runtime.model_catalog import ModelCatalog


class EmbedderConfig(BaseModel):
    task: InferenceTask = Field(
        default=InferenceTask.EMBEDDING,
        description="Inference task category for model catalog resolution.",
        json_schema_extra={"x-internal": True},
    )
    model_family: str = Field(
        "siglip2",
        description="Embedding backend family, for example 'siglip2' or 'clip'.",
    )
    size: ModelSize = Field(
        ModelSize.MEDIUM,
        description="Model size tier used when resolving the checkpoint from the catalog.",
    )
    device: DeviceTarget = Field(
        DeviceTarget.AUTO,
        description="Preferred execution device used to select the runtime and checkpoint.",
    )
    runtime: str | None = Field(
        default=None,
        description="Optional runtime override. Defaults to the catalog runtime for the selected device.",
        json_schema_extra={"x-internal": True},
    )
    artifact_path: str | None = Field(
        default=None,
        description="Optional local artifact path. When set, it overrides catalog checkpoint resolution.",
        json_schema_extra={"x-internal": True},
    )

class Embedder(ModelNode):
    InputPorts = {"image": "Image"}
    OutputPorts = {"embedding": "Embedding"}

    def __init__(self, node_id: str = "embedder", config: dict | None = None, **kwargs) -> None:
        raw_config = config or {}
        validated = EmbedderConfig.model_validate(raw_config)
        intent = ModelIntent(
            task=validated.task,
            model_family=validated.model_family,
            size=validated.size,
            device=validated.device,
        )
        resolved = ModelCatalog.resolve(intent)

        if "backend" not in kwargs:
            kwargs["backend"] = BackendRegistry.get(
                task=resolved.backend_task,
                model=resolved.backend_model,
            )

        resolved_config = validated.model_dump(mode="json", exclude_none=True)
        resolved_config.update(
            {
                "runtime": validated.runtime or resolved.runtime,
                "checkpoint_id": resolved.checkpoint_id,
                "backend_task": resolved.backend_task,
                "backend_model": resolved.backend_model,
            }
        )
        if hasattr(kwargs["backend"], "configure"):
            kwargs["backend"].configure(resolved_config)
        super().__init__(node_id=node_id, config=resolved_config, **kwargs)

    def preprocess(self, inputs, context):
        _ = context
        return inputs["image"].data

    def normalize_outputs(self, outputs):
        if not isinstance(outputs, dict):
            raise TypeError(f"{self.node_id}: backend output must be a dict.")
        if "embedding" not in outputs:
            raise TypeError(f"{self.node_id}: backend output must define 'embedding'.")
        embedding = outputs["embedding"]
        if hasattr(embedding, "model_dump"):
            embedding = embedding.model_dump()
        return {"embedding": Embedding.model_validate(embedding)}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return EmbedderConfig

    @classmethod
    def get_config_options(cls) -> dict[str, Any]:
        return {"model_catalog": ModelCatalog.list_options(task=InferenceTask.EMBEDDING)}
