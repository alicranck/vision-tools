"""
Training configuration schemas.

All training hyperparameters and settings are defined here as Pydantic models,
enabling validation, serialization, and LLM-composable configuration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class AugmentationPreset(str, Enum):
    """Predefined augmentation strategies."""
    NONE = "none"
    LIGHT = "light"       # Random flip, small rotation, brightness
    MODERATE = "moderate"  # + color jitter, scale variation, mosaic
    HEAVY = "heavy"        # + cutout, mixup, advanced geometric


class TrainConfig(BaseModel):
    """
    Configuration for tool fine-tuning.
    
    This is the primary configuration object passed to ToolTrainer.train().
    It covers hyperparameters, data handling, and checkpoint management.
    """
    # --- Core hyperparameters ---
    epochs: int = Field(50, ge=1, le=1000, description="Number of training epochs")
    batch_size: int = Field(16, ge=1, le=256, description="Training batch size")
    learning_rate: float = Field(0.01, gt=0.0, le=1.0, description="Initial learning rate")
    
    # --- Image processing ---
    imgsz: int = Field(640, ge=32, description="Training image size")
    
    # --- Data augmentation ---
    augmentation: AugmentationPreset = Field(
        AugmentationPreset.MODERATE,
        description="Augmentation preset to use"
    )
    
    # --- Checkpoints ---
    checkpoint_dir: Optional[str] = Field(
        None, description="Directory to save training checkpoints"
    )
    save_best: bool = Field(True, description="Save best checkpoint based on validation metric")
    save_every_n_epochs: Optional[int] = Field(
        None, description="Save checkpoint every N epochs (in addition to best)"
    )
    
    # --- Validation ---
    val_split: float = Field(
        0.2, ge=0.0, le=0.5,
        description="Fraction of dataset to use for validation (0 = no validation)"
    )
    
    # --- Device / resources ---
    device: Optional[str] = Field(None, description="Device override (cuda, cpu)")
    num_workers: int = Field(4, ge=0, description="DataLoader workers")
    
    # --- Framework-specific overrides ---
    extra_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed to the underlying training framework"
    )

    def to_framework_kwargs(self) -> Dict[str, Any]:
        """Convert to a flat dict suitable for passing to Ultralytics model.train()."""
        kwargs = {
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.imgsz,
            'lr0': self.learning_rate,
            'verbose': True,
        }
        if self.device:
            kwargs['device'] = self.device
        if self.checkpoint_dir:
            kwargs['project'] = self.checkpoint_dir
        kwargs.update(self.extra_args)
        return kwargs
