"""
Configuration models for vision-tools v2 pipelines.

These Pydantic models define the declarative configuration format that
LLM agents produce and the pipeline consumes.

Example pipeline config (JSON)::

    {
        "name": "restaurant_monitoring",
        "nodes": [
            {
                "node_id": "detector",
                "node_type": "object_detector",
                "config": {
                    "task": "detection",
                    "model_family": "yolo",
                    "size": "small",
                    "device": "cpu",
                    "vocabulary": ["person", "food"]
                }
            },
            {
                "node_id": "captioner",
                "node_type": "captioner",
                "config": {"model": "smolvlm"},
                "depends_on": ["detector"]
            }
        ],
        "execution": {"mode": "parallel", "max_workers": 4}
    }
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class NodeConfig(BaseModel):
    """Configuration for a single node in the pipeline DAG.

    Attributes:
        node_id: Unique identifier for this node within the pipeline.
        node_type: Registry key (e.g. "object_detector", "embedder").
        config: Node-specific parameters passed to the node constructor.
        depends_on: List of node IDs whose outputs this node consumes.
    """
    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="Registry key for the node type")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Node-specific configuration parameters",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Node IDs whose outputs feed into this node",
    )

    model_config = ConfigDict(extra="forbid")


class InferenceTask(str, Enum):
    """High-level inference task selected by the app/LLM."""
    DETECTION = "detection"
    OPEN_VOCAB_DETECTION = "open_vocab_detection"
    EMBEDDING = "embedding"
    CAPTIONING = "captioning"
    POSE = "pose"


class ModelSize(str, Enum):
    """Canonical model size tiers exposed to the app."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class DeviceTarget(str, Enum):
    """Execution target selected by the app/LLM."""
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class ModelIntent(BaseModel):
    """High-level model selection contract for app -> vision-tools.

    The app chooses task/family/size/device; vision-tools resolves concrete
    checkpoint/runtime internally.
    """
    task: InferenceTask
    model_family: str = Field(..., description="Model family (e.g. 'yolo', 'rtdetr').")
    size: ModelSize = Field(ModelSize.SMALL)
    device: DeviceTarget = Field(DeviceTarget.AUTO)

    model_config = ConfigDict(extra="forbid")


class ExecutionConfig(BaseModel):
    """Execution settings for a pipeline.

    Attributes:
        mode: Execution mode — "sequential" or "parallel".
        max_workers: Maximum number of parallel workers (for parallel mode).
        warmup_rounds: Number of warmup inference rounds per model node.
        verify_on_init: Whether to run verification on init.
    """
    mode: str = Field("sequential", pattern=r"^(sequential|parallel)$")
    max_workers: int = Field(4, ge=1, le=32)
    warmup_rounds: int = Field(4, ge=0)
    verify_on_init: bool = True

    model_config = ConfigDict(extra="forbid")


class PipelineConfig(BaseModel):
    """Full pipeline specification — the top-level config an LLM agent produces.

    Attributes:
        name: Human-readable name for the pipeline.
        nodes: List of node configurations defining the DAG.
        execution: Execution settings.
    """
    name: str = Field("pipeline", description="Pipeline name")
    nodes: list[NodeConfig] = Field(
        ...,
        min_length=1,
        description="Pipeline nodes (at least one required)",
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Pipeline execution settings",
    )

    model_config = ConfigDict(extra="forbid")
