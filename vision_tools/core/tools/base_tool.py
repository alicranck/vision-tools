from abc import ABC, abstractmethod
import traceback
import numpy as np
import torch
import logging
from PIL import Image

from ...utils.types import ImageHandle, List, Any, FrameContext
from ...utils.image_utils import load_image_opencv
from ...utils.locations import APP_DIR

logger = logging.getLogger(__name__)


class ToolKey:
    """
    A descriptor for a data key provided or required by a tool.
    This acts as a manifest entry for the tool's inputs and outputs.
    
    Attributes:
        key_name (str): The name of the key in the data dictionary.
        data_type (Any): The expected data type of the value.
        description (str): A brief description of what this data represents.
        required (bool): Whether this key is mandatory for the tool to function.
    """
    def __init__(self, key_name: str, data_type: Any, description: str,
                 required: bool = False):
        self.key_name = key_name
        self.data_type = data_type
        self.description = description
        self.required = required

    def __repr__(self):
        return f"ToolKey(key='{self.key_name}', type={self.data_type.__name__}, required={self.required})"


class BaseVisionTool(ABC):
    """
    Abstract Base Class for a modular vision tool.
    """
    def __init__(self, model_id: str, config: dict, 
                 device: str = 'cpu'):
        self.model_id : str = model_id
        self.device: str = device

        self.model : Any
        self.loaded: bool = False
        self.tool_name: str = self.__class__.__name__

        self.last_result: Any = None
        self.last_context: FrameContext = None
        
        # Trigger configuration
        self.trigger = config.get('trigger', {})
        logger.info(f"Trigger for {self.tool_name}: {self.trigger}")

        self.load_tool(config)

    def load_tool(self, config):
        """
        Public method to load, verify, and warmup the model.
        This is the common "init model" flow.
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
            self._warmup()
            self.loaded = True
            logger.info(f"{self.tool_name} successfully loaded and warmed up on {self.device}.")
            
        except Exception as e:
            self.model = None
            self.loaded = False
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
        self.loaded = False
        logger.info(f"{self.tool_name} unloaded and cleared from {self.device}.")

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

    def process(self, frame_handle: ImageHandle, data: dict, context: FrameContext = None) -> dict:
        """
        Public method to run the full inference pipeline.
        This is the common "process frame" flow.
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
        else:
            new_data = self.extrapolate_last(frame_handle)
        
        updated_data = {**data, **new_data}

        return updated_data
    
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

    def _configure(self, config: dict):
        """Child implements tool-specific configuration logic."""
        pass

    def _warmup(self):
        """
        Performs a dummy inference run to initialize the model on the device.
        This helps avoid latency spikes during the first real inference.
        """
        logger.info("Warming up model...")
        try:
            for _ in range(4):
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                inputs = self.preprocess(dummy_image)
                _ = self.inference(inputs)
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

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