"""
ToolTrainer — Orchestrates tool fine-tuning lifecycle.

Handles:
- Dataset validation for tool compatibility
- State machine transitions (BASE → TRAINING → TUNED)
- Progress callbacks
- Checkpoint saving
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from ..core.tools.base_tool import BaseVisionTool
from ..utils.schemas import ToolState
from .config import TrainConfig
from .dataset import VisionDataset

# Support new ModelNode if available
try:
    from ..nodes.model_node import ModelNode
except ImportError:
    ModelNode = None  # type: ignore

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Raised when training fails."""
    pass


class ToolTrainer:
    """
    Orchestrates the training lifecycle for vision tools.
    
    Validates dataset compatibility, manages state transitions,
    and delegates actual training to the tool's _train_impl().
    
    Usage:
        trainer = ToolTrainer(tool=detector)
        dataset = VisionDataset("path/to/data.yaml")
        config = TrainConfig(epochs=50, batch_size=16)
        await trainer.train(dataset, config)
    """

    def __init__(self, tool: BaseVisionTool,
                 on_progress: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Args:
            tool: The vision tool to train
            on_progress: Optional callback called with progress updates
        """
        self.tool = tool
        self.on_progress = on_progress
        self._training_active = False

    def _emit_progress(self, event: str, **kwargs):
        """Emit a progress event to the callback."""
        if self.on_progress:
            self.on_progress({
                'event': event,
                'tool': self.tool.tool_name,
                **kwargs,
            })

    async def train(self, dataset: VisionDataset, config: TrainConfig) -> None:
        """
        Run the full training pipeline:
        1. Validate dataset for tool compatibility
        2. Transition tool state to TRAINING
        3. Delegate to tool._train_impl()
        4. Transition to TUNED on success, rollback on failure
        
        Args:
            dataset: The dataset to train on
            config: Training configuration
            
        Raises:
            TrainingError: If validation fails or training errors out
        """
        if self._training_active:
            raise TrainingError("Training is already in progress for this tool")
        
        # 1. Validate dataset compatibility
        tool_type = self._get_tool_type()
        errors = dataset.validate_for_tool(tool_type)
        if errors:
            raise TrainingError(
                f"Dataset validation failed for {self.tool.tool_name}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        
        # 2. Prepare training kwargs
        dataset_ref = dataset.get_ref()
        train_kwargs = config.to_framework_kwargs()
        train_kwargs['data'] = dataset_ref
        
        logger.info(
            f"Starting training for {self.tool.tool_name} on "
            f"{dataset.name} ({dataset.num_images} images, {config.epochs} epochs)"
        )
        self._emit_progress('training_started', dataset=dataset.name,
                           epochs=config.epochs)
        
        # 3. Train (tool handles state transitions)
        self._training_active = True
        try:
            await self.tool.train(dataset_ref, train_kwargs)
            self._emit_progress('training_completed', state=self.tool.state.value)
            logger.info(
                f"Training complete for {self.tool.tool_name}. "
                f"State: {self.tool.state.value}"
            )
        except Exception as e:
            self._emit_progress('training_failed', error=str(e))
            raise TrainingError(f"Training failed: {e}") from e
        finally:
            self._training_active = False

    def _get_tool_type(self) -> str:
        """Infer tool type string from the tool class for dataset validation."""
        from ..core.tools.pipeline import AVAILABLE_TOOL_TYPES
        for type_name, type_class in AVAILABLE_TOOL_TYPES.items():
            if isinstance(self.tool, type_class):
                return type_name
        return 'unknown'
