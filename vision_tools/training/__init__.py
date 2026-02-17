"""
Vision Tools Training Infrastructure.

Provides modular training components:
- TrainConfig: Pydantic training configuration
- VisionDataset: Lazy-loading dataset with COCO-format support
- ToolTrainer: Orchestrates training lifecycle
"""
from .config import TrainConfig
from .dataset import VisionDataset
from .trainer import ToolTrainer

__all__ = ['TrainConfig', 'VisionDataset', 'ToolTrainer']
