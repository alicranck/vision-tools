import os
import yaml
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from .detection import OpenVocabularyDetector
from .captioning import LlamaCppCaptioner
from .pose_estimation import PoseEstimator
from .embedder import CLIPEmbedder, JinaEmbedder, SigLIP2Embedder
from ...utils.types import FrameContext


AVAILABLE_TOOL_TYPES = {
    'ov_detection': OpenVocabularyDetector,
    'captioning': LlamaCppCaptioner,
    'pose_estimation': PoseEstimator,
    'embedding': SigLIP2Embedder # Switched to SigLIP2Embedder
}


class PipelineConfig(BaseModel):
    """Defines the expected structure for the JSON configuration body.
    In the future, this will be represented as layers of a DAG to allow for more complex pipelines."""    
    tool_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Nested dictionary of tool-specific static configuration " \
                        "(e.g., {'ov_detection': {'vocabulary': ['person', 'car']}})."
    )


class VisionPipeline:
    """A modular vision processing pipeline that sequentially 
    applies a series of tools to input frames."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tools = self._initialize_tools()

    def run_pipeline(self, frame: Any, context: FrameContext = None) -> Dict[str, Any]:
        """
        Executes the pipeline on a single frame.
        Iterates through all initialized tools and aggregates their results.
        
        Args:
            frame (Any): The input video frame (usually a numpy array).
            context (FrameContext, optional): Metadata about the current frame (index, timestamp, etc.).
            
        Returns:
            Dict[str, Any]: A dictionary containing the original frame and the aggregated results from all tools.
        """
        data = {"tools_run": False}
        for tool in self.tools:
            tool_results, tool_run = tool.process(frame, data, context=context)
            data.update(tool_results)
            data["tools_run"] = data["tools_run"] or tool_run
            
        return frame, data

    def extrapolate_last(self, frame: Any) -> Dict[str, Any]:
        
        data = {}
        for tool in self.tools:
            tool_results = tool.extrapolate_last(frame)
            data.update(tool_results)

        return frame, data

    def _initialize_tools(self) -> List[Any]:

        tools = []
        for tool_type in self.config.tool_settings.keys():

            if tool_type not in AVAILABLE_TOOL_TYPES:
                raise ValueError(f"Tool type '{tool_type}' is not recognized.")
            
            tool_class = AVAILABLE_TOOL_TYPES[tool_type]
            tool_config = self.config.tool_settings.get(tool_type, {})
            tool_base_config = _get_base_tool_config(tool_type)
            tool_base_config.update(tool_config)

            tool_instance = tool_class(model_id=tool_base_config.pop('model'), 
                                                            config=tool_base_config)
            tools.append(tool_instance)
        
        return tools

    def unload_tools(self):
        for tool in self.tools:
            tool.unload_tool()


def _get_base_tool_config(tool_type: str) -> Dict[str, Any]:
    configs_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    config_path = os.path.join(configs_dir, f'{tool_type}.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


        
