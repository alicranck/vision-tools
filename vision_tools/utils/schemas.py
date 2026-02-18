"""
Backward-compatibility re-export.

All schemas have moved to ``vision_tools.core.schemas``.
This module re-exports them so existing code continues to work.

.. deprecated:: 2.0
    Import from ``vision_tools.core.schemas`` instead.
"""
from pydantic import BaseModel, Field

from vision_tools.core.schemas import (  # noqa: F401
    ToolState,
    ModelScale,
    BoundingBox,
    DetectionResult,
    SegmentationMask,
    SegmentationResult,
    Embedding,
    EmbeddingResult,
    Keypoint,
    PoseKeypoints,
    PoseResult,
    Caption,
    CaptionResult,
    FrameMetadata,
    FrameResult,
    BatchPayload,
)


# Deprecated stub — kept for backward compat with BaseVisionTool (removed in Stage 6)
class ToolIOContract(BaseModel):
    """Deprecated: use Node.InputSchema / OutputSchema instead."""
    key: str = Field(..., description="Key name in FrameResult.results")
    schema_type: str = Field(..., description="Fully qualified schema class name")
    required: bool = Field(False, description="Whether this input is mandatory")
    description: str = Field("", description="Human-readable description")

