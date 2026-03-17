from __future__ import annotations

from pydantic import BaseModel, Field

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.config import DeviceTarget, InferenceTask, ModelIntent, ModelSize
from vision_tools.core.graph_types import Caption
from vision_tools.nodes.model.model_node import ModelNode
from vision_tools.runtime.model_catalog import ModelCatalog


class CaptionerConfig(BaseModel):
    task: InferenceTask = Field(
        default=InferenceTask.CAPTIONING,
        description="Inference task category for model catalog resolution.",
    )
    model_family: str = Field(
        "smolvlm",
        description="Captioning backend family, for example 'smolvlm' or 'llamacpp'.",
    )
    size: ModelSize = Field(
        ModelSize.SMALL,
        description="Model size tier used when resolving the checkpoint from the catalog.",
    )
    device: DeviceTarget = Field(
        DeviceTarget.AUTO,
        description="Preferred execution device used to select the runtime and checkpoint.",
    )
    runtime: str | None = Field(
        default=None,
        description="Optional runtime override. Defaults to the catalog runtime for the selected device.",
    )
    imgsz: int = Field(512, description="Input image size for preprocessing.")
    max_tokens: int = Field(64, description="Maximum number of tokens to generate.")
    artifact_path: str | None = Field(
        default=None,
        description="Optional local artifact path. When set, it overrides catalog checkpoint resolution.",
    )

class Captioner(ModelNode):
    InputPorts = {"image": "Image"}
    OutputPorts = {"caption": "Caption"}

    def __init__(self, node_id: str = "captioner", config: dict | None = None, **kwargs) -> None:
        validated = CaptionerConfig.model_validate(config or {})
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
        if "caption" not in outputs:
            raise TypeError(f"{self.node_id}: backend output must define 'caption'.")
        caption = outputs["caption"]
        if hasattr(caption, "model_dump"):
            caption = caption.model_dump()
        return {"caption": Caption.model_validate(caption)}

    @classmethod
    def get_config_schema(cls) -> type[BaseModel]:
        return CaptionerConfig

    @classmethod
    def get_config_options(cls) -> dict[str, Any]:
        return {"model_catalog": ModelCatalog.list_options(task=InferenceTask.CAPTIONING)}
