"""
BaseVisionTool — Abstract base class for all vision tools in the VisionPilot framework.

Provides:
- Lifecycle management (load / unload / state machine)
- Typed I/O contracts via Pydantic schemas
- Single-frame and batch processing
- Training interface with state transitions
- Resource-aware model selection and device detection
- Trigger logic for frame-skipping (stride, scene change, time)
"""
from abc import ABC, abstractmethod
import traceback
import os
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np
import torch
import logging
from PIL import Image
from pydantic import BaseModel

from ...utils.types import ImageHandle, Any, FrameContext
from ...utils.image_utils import load_image_opencv
from ...utils.locations import APP_DIR, CACHE_DIR
from ...utils.resource_monitor import get_system_resources, SystemResources
from ...utils.schemas import (
    ToolState, ModelScale, FrameMetadata, FrameResult,
    ToolIOContract,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy ToolKey — kept for backward compat during migration
# ---------------------------------------------------------------------------

class ToolKey:
    """
    A descriptor for a data key provided or required by a tool.
    
    .. deprecated:: Use ToolIOContract + Pydantic OutputSchema instead.
    """
    def __init__(self, key_name: str, data_type: Any, description: str,
                 required: bool = False):
        self.key_name = key_name
        self.data_type = data_type
        self.description = description
        self.required = required

    def __repr__(self):
        return f"ToolKey(key='{self.key_name}', type={getattr(self.data_type, '__name__', self.data_type)}, required={self.required})"


# ---------------------------------------------------------------------------
# BaseVisionTool
# ---------------------------------------------------------------------------

class BaseVisionTool(ABC):
    """
    Abstract Base Class for a modular vision tool.
    
    Every concrete tool (detector, embedder, captioner, etc.) inherits from this
    and implements the abstract methods for its specific model.
    
    Args:
        model_id: Model identifier or path (override, or use config-based selection if None)
        config: Tool configuration dict (may contain 'models' schema for variant selection)
        device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        mode: Performance mode ('speed', 'balanced', 'accuracy', 'auto')
    """
    # Valid mode options
    VALID_MODES = ('speed', 'balanced', 'accuracy', 'auto')

    # --- Typed I/O contracts (override in subclasses) ---
    # Subclasses should set these to Pydantic BaseModel subclasses
    # to enable pipeline validation. E.g.:
    #   OutputSchema = DetectionResult
    OutputSchema: Optional[Type[BaseModel]] = None
    InputSchema: Optional[Type[BaseModel]] = None

    def __init__(self, model_id: str = None, config: dict = None, 
                 device: str = None, mode: str = 'auto'):
        if config is None:
            config = {}
            
        # Validate mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}")
        self.mode = mode
        
        # Detect system resources for scaling
        self.resources: SystemResources = get_system_resources()
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if self.resources.has_gpu else "cpu"
        else:
            self.device = device

        # Set tool identity early (needed by _resolve_model_from_config)
        self.model: Any = None
        self.tool_name: str = self.__class__.__name__

        # Resolve model_id from config if not provided explicitly
        if model_id is None:
            self.model_id = self._resolve_model_from_config(config)
        else:
            self.model_id = model_id

        # --- State machine ---
        self._state: ToolState = ToolState.UNTRAINED

        self.last_result: Any = None
        self.last_context: FrameContext = None
        
        # Trigger configuration
        self.trigger = config.get('trigger', {})
        logger.info(f"Trigger for {self.tool_name}: {self.trigger}")

        self.load_tool(config)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    @property
    def state(self) -> ToolState:
        return self._state

    @state.setter
    def state(self, new_state: ToolState):
        if not self._state.can_transition_to(new_state):
            raise ValueError(
                f"{self.tool_name}: Invalid state transition "
                f"{self._state.value} → {new_state.value}"
            )
        old = self._state
        self._state = new_state
        logger.info(f"{self.tool_name}: state {old.value} → {new_state.value}")

    @property
    def loaded(self) -> bool:
        """Backward-compatible: tool is loaded if state is BASE or TUNED."""
        return self._state in (ToolState.BASE, ToolState.TUNED)

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _resolve_model_from_config(self, config: dict) -> str:
        """
        Select model variant based on resources and mode preference.
        
        Config schema (optional):
            models:
              speed: {id: "model-small", min_vram_gb: 0}
              balanced: {id: "model-medium", min_vram_gb: 4}
              accuracy: {id: "model-large", min_vram_gb: 8}
            model: "fallback-model"  # Used if models not defined
        """
        models = config.get('models')
        
        if not models:
            # Legacy config: single 'model' key
            model = config.get('model')
            if model is None:
                raise ValueError(f"{self.tool_name}: No 'model' or 'models' found in config")
            return model
        
        if self.mode == 'auto':
            # Pick best that fits in available VRAM
            vram = self.resources.gpu_vram_gb or 0
            for tier in ['accuracy', 'balanced', 'speed']:
                if tier in models:
                    min_vram = models[tier].get('min_vram_gb', 0)
                    if vram >= min_vram:
                        logger.info(f"{self.tool_name}: Auto-selected '{tier}' model (VRAM: {vram:.1f}GB)")
                        return models[tier]['id']
            # Fallback to speed if nothing fits
            return models.get('speed', {}).get('id', config.get('model'))
        else:
            # Use explicit mode preference
            if self.mode in models:
                logger.info(f"{self.tool_name}: Using '{self.mode}' model")
                return models[self.mode]['id']
            else:
                fallback = models.get('balanced') or next(iter(models.values()))
                logger.warning(f"{self.tool_name}: Mode '{self.mode}' not in config, using fallback")
                return fallback['id']

    def _resolve_model_path(self, model_identifier: str) -> str:
        """
        Resolves the local path for a model.
        1. Checks if it's an existing file or absolute path.
        2. Checks if it's in the cache.
        3. Attempts to download it if missing.
        """
        # 1. Existing path check
        if os.path.isfile(model_identifier):
            return model_identifier
        
        # 2. Check Cache
        if not CACHE_DIR.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
        filename = Path(model_identifier).name
        cached_path = CACHE_DIR / filename
        
        if cached_path.exists():
            logger.info(f"Found model in cache: {cached_path}")
            return str(cached_path)
            
        # 3. Download
        print("Model not found in cache. downloading...", cached_path)
        logger.info(f"Model {filename} not found in cache. downloading...")
        try:
            logger.info(f"Downloading {model_identifier} to {cached_path}")
            downloaded_path = self.download_ckpt(model_identifier, cached_path)
            if downloaded_path and os.path.exists(downloaded_path):
                return str(downloaded_path)
        except NotImplementedError:
             # Fallback for tools that rely on libraries (like transformers) to handle downloads
             logger.info(f"{self.tool_name} does not implement manual download, passing identifier through.")
             return model_identifier
        except Exception as e:
            logger.error(f"Failed to download model {model_identifier}: {e}")
            
        # Return identifier if resolution fails (library might handle it)
        return model_identifier

    def download_ckpt(self, model_id: str, destination: Path) -> Path:
        """
        Downloads the checkpoint to the destination.
        Should be implemented by tools that require manual file handling.
        """
        raise NotImplementedError("This tool does not implement manual checkpoint download.")

    # ------------------------------------------------------------------
    # Lifecycle: load / unload
    # ------------------------------------------------------------------

    def load_tool(self, config):
        """
        Public method to load and verify the model.
        Does NOT warm up — call warmup() separately if needed.
        """
        if self.loaded:
            logger.info(f"{self.tool_name} is already loaded.")
            return

        logger.info(f"Loading {self.tool_name} with model: {self.model_id}...")

        for key in self.config_keys:
            if key.required:
                if key.key_name not in config:
                    raise ValueError(f"ERROR: Missing required config key '{key.key_name}' for {self.tool_name}.")
                
        self._configure(config)

        try:
            self.model = self._load_model()
            # Transition to BASE (pretrained) state
            self._state = ToolState.BASE
            logger.info(f"{self.tool_name} successfully loaded on {self.device}.")
            
        except Exception as e:
            self.model = None
            self._state = ToolState.UNTRAINED
            logger.error(f"Failed to load {self.tool_name}. Error: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def unload_tool(self):
        """
        Public method to clear the model from device memory.
        This is the common "teardown" flow.
        """
        if self.model:
            del self.model
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        self.model = None
        self._state = ToolState.UNTRAINED
        logger.info(f"{self.tool_name} unloaded and cleared from {self.device}.")

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    def should_run(self, context: FrameContext) -> bool:
        """
        Determines if the tool should run based on the current frame context and trigger settings.
        """
        if self.last_result is None or not context:
            return True # Always run if we have no history
            
        trigger_type = self.trigger.get('type', 'always')
        
        if trigger_type == 'stride':
            stride = self.trigger.get('value', 1)
            return (context.frame_idx % stride) == 0
            
        elif trigger_type == 'scene_change':
            threshold = self.trigger.get('threshold', 0.3)
            return context.scene_change_score >= threshold

        elif trigger_type == 'time':
            interval = self.trigger.get('value', 1)
            return (context.timestamp - self.last_context.timestamp) >= interval
            
        return True

    # ------------------------------------------------------------------
    # Single-frame processing (existing interface)
    # ------------------------------------------------------------------

    def process(self, frame_handle: ImageHandle, data: dict, context: FrameContext = None) -> dict:
        """
        Public method to run the full inference pipeline on a single frame.
        """
        if not self.loaded:
            raise RuntimeError(f"ERROR: {self.tool_name} is not loaded. Call .load_tool() first.")

        if self.should_run(context):
            logger.debug(f"Trigger detected for {self.tool_name}")
            frame = load_image_opencv(frame_handle)
            model_input = self.preprocess(frame)
            
            with torch.no_grad():
                raw_output = self.inference(model_input)

            self.last_result = raw_output
            self.last_context = context
            
            new_data = self.postprocess(raw_output, frame.shape)
            did_run = True
        else:
            new_data = self.extrapolate_last(frame_handle)
            did_run = False
        
        updated_data = {**data, **new_data}

        return updated_data, did_run

    # ------------------------------------------------------------------
    # Batch processing (new VisionPilot interface)
    # ------------------------------------------------------------------

    def process_batch(self, frames: List[np.ndarray],
                      contexts: List[FrameContext] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of frames. Default implementation loops single-frame
        processing. Subclasses can override for true batched inference.
        
        Args:
            frames: List of raw numpy frames.
            contexts: Optional list of FrameContext per frame.
            
        Returns:
            List of result dicts, one per frame.
        """
        if not self.loaded:
            raise RuntimeError(f"{self.tool_name} is not loaded.")
        
        if contexts is None:
            contexts = [None] * len(frames)
        
        results = []
        for frame, ctx in zip(frames, contexts):
            data, _ = self.process(frame, {}, ctx)
            results.append(data)
        return results

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    async def train(self, dataset_ref: str, config: Optional[dict] = None) -> None:
        """
        Fine-tune the model on a user-provided dataset.
        
        Transitions state: current → TRAINING → TUNED (or rollback on failure).
        
        Args:
            dataset_ref: Path or reference to the training dataset.
            config: Training hyperparameters (epochs, lr, batch_size, etc.)
        
        Subclasses that support training must override _train_impl().
        """
        if config is None:
            config = {}
        
        previous_state = self._state
        
        try:
            self.state = ToolState.TRAINING
            logger.info(f"{self.tool_name}: Starting training on {dataset_ref}")
            
            await self._train_impl(dataset_ref, config)
            
            self.state = ToolState.TUNED
            logger.info(f"{self.tool_name}: Training complete. State → TUNED")
            
        except Exception as e:
            logger.error(f"{self.tool_name}: Training failed: {e}")
            # Rollback state
            self._state = previous_state
            raise

    async def _train_impl(self, dataset_ref: str, config: dict) -> None:
        """
        Subclass implements the actual training logic.
        Override this — not train() — to add training support.
        """
        raise NotImplementedError(
            f"{self.tool_name} does not support training. "
            "Override _train_impl() to add training support."
        )

    # ------------------------------------------------------------------
    # Extrapolation
    # ------------------------------------------------------------------

    def extrapolate_last(self, frame_handle: ImageHandle) -> Any:
        """
        Public method to return the last inference result with some extrapolation logic
        if applicable.
        """
        if self.last_result is None:
            raise RuntimeError(f"ERROR: No previous result available in {self.tool_name}.")
        
        frame = load_image_opencv(frame_handle)
        updated_data = self.postprocess(self.last_result, frame.shape)

        return updated_data

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    def _configure(self, config: dict):
        """Child implements tool-specific configuration logic."""
        pass

    def warmup(self, rounds: int = 4):
        """
        Performs dummy inference runs to initialize the model on the device.
        This helps avoid latency spikes during the first real inference.

        Call this explicitly after load_tool() when you need warm caches.
        VisionPipeline calls this automatically after constructing all tools.
        """
        logger.info(f"Warming up {self.tool_name} ({rounds} rounds)...")
        try:
            for _ in range(rounds):
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                inputs = self.preprocess(dummy_image)
                _ = self.inference(inputs)
            logger.info(f"{self.tool_name} warmup complete.")
        except Exception as e:
            logger.warning(f"{self.tool_name} warmup failed: {e}")

    def preprocess(self, frame: np.ndarray) -> Any:
        """Child implements frame-to-tensor logic (resize, normalize, to-device)."""
        return frame
    
    @abstractmethod
    def _load_model(self) -> Any:
        """Child implements the specific model loading logic (e.g., YOLO(path))."""
        pass

    @abstractmethod
    def inference(self, model_inputs: Any) -> Any:
        """Child implements the raw model.forward() call."""
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        """Child implements logic to parse raw_output."""
        pass

    @property
    @abstractmethod
    def output_keys(self) -> List[ToolKey]:
        """
        Class property declaring the data keys this tool *produces* and adds to the 'data' dictionary.
        """
        pass

    @property
    @abstractmethod
    def processing_input_keys(self) -> List[ToolKey]:
        """
        Class property declaring the data keys this tool *requires* to be present in the 'data' dictionary to run.
        """
        pass

    @property
    @abstractmethod
    def config_keys(self) -> List[ToolKey]:
        """
        Class property declaring the configuration keys this tool *requires* to be present in the 'config' dictionary to run.
        """
        pass