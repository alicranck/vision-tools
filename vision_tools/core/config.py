"""
Configuration models for the canonical typed-port pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NodeConfig(BaseModel):
    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="Registry key for the node type")
    config: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="Input port name -> '<node_id>.<port_name>' binding",
    )

    model_config = ConfigDict(extra="forbid")


class InferenceTask(str, Enum):
    DETECTION = "detection"
    OPEN_VOCAB_DETECTION = "open_vocab_detection"
    EMBEDDING = "embedding"
    CAPTIONING = "captioning"
    POSE = "pose"


class ModelSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class DeviceTarget(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class ModelIntent(BaseModel):
    task: InferenceTask
    model_family: str
    size: ModelSize = Field(ModelSize.SMALL)
    device: DeviceTarget = Field(DeviceTarget.AUTO)

    model_config = ConfigDict(extra="forbid")


class ExecutionConfig(BaseModel):
    mode: str = Field("sequential", pattern=r"^sequential$")
    max_workers: int = Field(1, ge=1, le=32)
    warmup_rounds: int = Field(0, ge=0)
    verify_on_init: bool = False

    model_config = ConfigDict(extra="forbid")


class PipelineConfig(BaseModel):
    name: str = Field("pipeline")
    nodes: list[NodeConfig] = Field(..., min_length=1)
    outputs: dict[str, str] = Field(default_factory=dict)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    model_config = ConfigDict(extra="forbid")
