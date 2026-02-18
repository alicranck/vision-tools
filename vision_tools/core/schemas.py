"""
Core Pydantic data contracts for VisionPilot.

All tool outputs must conform to these schemas. The Pipeline validates
tool I/O compatibility at construction time using these types.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ToolState(str, Enum):
    """Lifecycle state of a VisionTool."""
    UNTRAINED = "untrained"  # Random / no weights
    BASE = "base"            # Pretrained checkpoint, usable out-of-the-box
    TRAINING = "training"    # Currently being fine-tuned
    TUNED = "tuned"          # Fine-tuned for a specific task / user data

    # Valid transitions
    @classmethod
    def valid_transitions(cls) -> Dict["ToolState", List["ToolState"]]:
        return {
            cls.UNTRAINED: [cls.TRAINING],
            cls.BASE: [cls.TRAINING, cls.TUNED],
            cls.TRAINING: [cls.TUNED, cls.BASE, cls.UNTRAINED],  # failure → rollback
            cls.TUNED: [cls.TRAINING],  # re-train
        }

    def can_transition_to(self, target: "ToolState") -> bool:
        return target in self.valid_transitions().get(self, [])


class ModelScale(str, Enum):
    """Model size tier for hardware-adaptive selection."""
    NANO = "nano"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"


# ---------------------------------------------------------------------------
# Detection schemas
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """A single detected bounding box."""
    xyxy: List[float] = Field(..., min_length=4, max_length=4,
                               description="Bounding box coordinates [x1, y1, x2, y2]")
    class_id: int = Field(..., description="Class index")
    confidence: float = Field(..., ge=0.0, le=1.0)
    tracker_id: Optional[int] = Field(None, description="Track ID from tracker")
    class_name: Optional[str] = Field(None, description="Human-readable class name")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DetectionResult(BaseModel):
    """Output schema for detection tools."""
    boxes: List[BoundingBox] = Field(default_factory=list)
    class_names: Optional[Dict[int, str]] = Field(None, description="Class ID → name mapping")


# ---------------------------------------------------------------------------
# Segmentation schemas
# ---------------------------------------------------------------------------

class SegmentationMask(BaseModel):
    """A single segmentation mask using Run-Length Encoding for efficiency."""
    rle_counts: List[int] = Field(..., description="RLE-encoded mask counts")
    height: int = Field(..., gt=0)
    width: int = Field(..., gt=0)
    class_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: Optional[str] = None

    def to_binary_mask(self) -> np.ndarray:
        """Decode RLE to a binary numpy mask."""
        mask = np.zeros(self.height * self.width, dtype=np.uint8)
        pos = 0
        for i, count in enumerate(self.rle_counts):
            if i % 2 == 1:  # odd indices are foreground runs
                mask[pos:pos + count] = 1
            pos += count
        return mask.reshape(self.height, self.width)

    @classmethod
    def from_binary_mask(cls, mask: np.ndarray, class_id: int,
                          confidence: float, class_name: Optional[str] = None) -> "SegmentationMask":
        """Encode a binary numpy mask to RLE.
        
        Convention: counts alternate [background, foreground, background, ...].
        Even indices (0, 2, 4...) are background runs, odd indices are foreground.
        """
        flat = mask.flatten().astype(np.uint8)
        n = len(flat)
        
        if n == 0:
            return cls(rle_counts=[], height=mask.shape[0], width=mask.shape[1],
                       class_id=class_id, confidence=confidence, class_name=class_name)
        
        # Build runs: always start with a background count (may be 0)
        counts: List[int] = []
        current_val = 0  # always start counting background
        run_len = 0
        
        for pixel in flat:
            if pixel == current_val:
                run_len += 1
            else:
                counts.append(run_len)
                current_val = pixel
                run_len = 1
        counts.append(run_len)
        
        # If mask starts with foreground, prepend a zero-length background run
        if flat[0] == 1:
            counts.insert(0, 0)
        
        return cls(
            rle_counts=counts,
            height=mask.shape[0],
            width=mask.shape[1],
            class_id=class_id,
            confidence=confidence,
            class_name=class_name,
        )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SegmentationResult(BaseModel):
    """Output schema for segmentation tools."""
    masks: List[SegmentationMask] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Embedding schemas
# ---------------------------------------------------------------------------

class Embedding(BaseModel):
    """A vector embedding produced by an embedding tool."""
    vector: List[float] = Field(..., description="Embedding vector")
    model_id: str = Field(..., description="Model that produced this embedding")
    dimension: int = Field(..., gt=0, description="Embedding dimensionality")


class EmbeddingResult(BaseModel):
    """Output schema for embedding tools."""
    embedding: Embedding


# ---------------------------------------------------------------------------
# Pose schemas
# ---------------------------------------------------------------------------

class Keypoint(BaseModel):
    """A single pose keypoint."""
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)


class PoseKeypoints(BaseModel):
    """Keypoints for a single detected person."""
    person_id: int
    keypoints: List[Keypoint] = Field(default_factory=list)


class PoseResult(BaseModel):
    """Output schema for pose estimation tools."""
    poses: List[PoseKeypoints] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Caption schemas
# ---------------------------------------------------------------------------

class Caption(BaseModel):
    """A text caption/description of an image."""
    text: str
    model_id: str


class CaptionResult(BaseModel):
    """Output schema for captioning tools."""
    caption: Caption


# ---------------------------------------------------------------------------
# Frame-level containers
# ---------------------------------------------------------------------------

class FrameMetadata(BaseModel):
    """Metadata associated with a single video frame."""
    frame_idx: int = Field(..., ge=0)
    timestamp: float = Field(0.0, ge=0.0, description="Seconds from video start")
    scene_change_score: float = Field(0.0, ge=0.0, le=1.0)
    camera_id: str = Field("default", description="Source camera identifier")


class FrameResult(BaseModel):
    """Aggregated results for a single processed frame."""
    metadata: FrameMetadata
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool name → tool output (validated Pydantic model serialized to dict)"
    )
    tools_run: bool = Field(False, description="Whether any tools actually ran on this frame")

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Batch-level container
# ---------------------------------------------------------------------------

class BatchPayload(BaseModel):
    """
    A batch of frames flowing through the pipeline.

    The `frames` field holds raw numpy arrays (excluded from serialization).
    Each tool receives this payload, processes the batch, and writes results
    into `frame_results`.
    """
    batch_id: str = Field(..., description="Unique batch identifier")
    frames: List[Any] = Field(default_factory=list,
                               description="Raw frames (numpy arrays), excluded from serialization")
    frame_metadatas: List[FrameMetadata] = Field(default_factory=list)
    frame_results: List[FrameResult] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __len__(self) -> int:
        return len(self.frames)


# ToolIOContract has been removed in v2 — replaced by Node.InputSchema / OutputSchema
