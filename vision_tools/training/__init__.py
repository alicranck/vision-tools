"""
Vision Tools Training Infrastructure.

Provides modular training components:
- TrainConfig: Pydantic training configuration
- VisionDataset: Lazy-loading dataset with COCO-format support
- ToolTrainer: Orchestrates training lifecycle
- Annotation / AnnotatedFrame: Unified polygon-based annotation model
"""
from .annotations import Annotation, AnnotatedFrame
from .artifacts import TrainedArtifact
from .config import TrainConfig
from .dataset import VisionDataset
from .trainer import ToolTrainer

__all__ = [
    'Annotation',
    'AnnotatedFrame',
    'TrainedArtifact',
    'TrainConfig',
    'VisionDataset',
    'ToolTrainer',
]
