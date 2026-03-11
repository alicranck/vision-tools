from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainedArtifact(BaseModel):
    """Artifact metadata returned from backend training."""

    artifact_path: str = Field(..., description="Filesystem path to the trained weights.")
    backend_task: str
    backend_model: str
    task: str
    model_family: str
    base_checkpoint_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
