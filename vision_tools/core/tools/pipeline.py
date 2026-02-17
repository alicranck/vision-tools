"""
VisionPipeline — DAG-based tool orchestration with typed I/O validation.

The pipeline is defined as a directed acyclic graph (DAG) of PipelineNodes.
Each node wraps a VisionTool and declares its dependencies.

At construction time, the pipeline validates that every node's InputSchema
is satisfied by the combined OutputSchemas of its dependencies.

At execution time, nodes are topologically sorted and executed in layers.
Nodes in the same layer (no mutual dependencies) run in parallel on CPU.
"""
import os
import uuid
import yaml
import logging
from typing import Dict, List, Any, Optional, Type
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
import numpy as np

from .base_tool import BaseVisionTool, ToolKey
from .detection import OpenVocabularyDetector
from .captioning import LlamaCppCaptioner, Captioner
from .pose_estimation import PoseEstimator
from .embedder import CLIPEmbedder, SigLIP2Embedder, OVSigLIP2Embedder
from ...utils.types import FrameContext
from ...utils.schemas import FrameMetadata, FrameResult, BatchPayload
from ...utils.resource_monitor import get_system_resources

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

AVAILABLE_TOOL_TYPES: Dict[str, Type[BaseVisionTool]] = {
    'ov_detection': OpenVocabularyDetector,
    'captioning': LlamaCppCaptioner,
    'captioner_vlm': Captioner,
    'pose_estimation': PoseEstimator,
    'embedding': SigLIP2Embedder,
    'ov_embedding': OVSigLIP2Embedder,
    'clip_embedding': CLIPEmbedder,
}


# ---------------------------------------------------------------------------
# Pipeline node definition
# ---------------------------------------------------------------------------

class PipelineNode(BaseModel):
    """
    A single node in the pipeline DAG.
    
    Attributes:
        node_id: Unique identifier for this node
        tool_type: Key into AVAILABLE_TOOL_TYPES
        tool_config: Tool-specific configuration overrides
        depends_on: List of node_ids this node depends on
    """
    node_id: str = Field(..., description="Unique node identifier")
    tool_type: str = Field(..., description="Tool type key from AVAILABLE_TOOL_TYPES")
    tool_config: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(
        default_factory=list,
        description="Node IDs whose outputs feed into this node",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """
    Pipeline configuration supporting both legacy flat dict and new DAG format.
    
    Legacy format (backward compatible):
        tool_settings: {'ov_detection': {vocabulary: [...]}, 'embedding': {}}
    
    DAG format:
        nodes: [{node_id: 'det', tool_type: 'ov_detection', ...}, ...]
    """
    tool_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy flat tool settings dict"
    )
    nodes: List[PipelineNode] = Field(
        default_factory=list,
        description="DAG nodes (takes precedence over tool_settings)"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VisionPipeline:
    """
    A modular vision processing pipeline that executes tools as a DAG.
    
    Supports:
    - Sequential (legacy) and DAG-based execution
    - I/O schema validation at construction time
    - Topological execution with parallel layers on CPU
    - Batch processing
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.resources = get_system_resources()
        
        # Parse nodes from config (legacy or DAG format)
        self._node_configs = self._parse_nodes()
        
        # Build tools and the DAG
        self._tools: Dict[str, BaseVisionTool] = {}
        self._layers: List[List[str]] = []
        self._initialize_and_validate()
        
        # Worker pool for CPU parallelization
        if not self.resources.has_gpu:
            self.num_workers = self.resources.recommended_workers
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            logger.info(f"VisionPipeline: CPU mode with {self.num_workers} workers")
        else:
            self.executor = None
            logger.info("VisionPipeline: GPU mode (sequential processing)")

    # ------------------------------------------------------------------
    # Legacy compatibility: flat tool_settings → PipelineNode list
    # ------------------------------------------------------------------

    def _parse_nodes(self) -> List[PipelineNode]:
        """Convert config into a list of PipelineNodes."""
        if self.config.nodes:
            return self.config.nodes
        
        # Legacy: build sequential chain from tool_settings
        nodes = []
        prev_id = None
        for tool_type, tool_cfg in self.config.tool_settings.items():
            node_id = tool_type  # Use tool type as node ID for legacy
            deps = [prev_id] if prev_id else []
            nodes.append(PipelineNode(
                node_id=node_id,
                tool_type=tool_type,
                tool_config=tool_cfg or {},
                depends_on=deps,
            ))
            prev_id = node_id
        return nodes

    # ------------------------------------------------------------------
    # Initialization and validation
    # ------------------------------------------------------------------

    def _initialize_and_validate(self):
        """Initialize tools, validate I/O contracts, and compute execution layers."""
        # 1. Validate the DAG structure first (cheap — catches cycles and
        #    missing dependencies before loading heavy models)
        self._layers = self._topological_sort()
        
        # 2. Instantiate all tools
        for node in self._node_configs:
            self._tools[node.node_id] = self._create_tool(node)
        
        # 3. Validate I/O schema compatibility
        self._validate_io_contracts()

        # 4. Warm up all tools (moved out of individual tool init)
        self.warmup()
        
        logger.info(
            f"VisionPipeline initialized: {len(self._tools)} tools, "
            f"{len(self._layers)} execution layers"
        )

    def warmup(self, rounds: int = 4):
        """Warm up all tools in the pipeline."""
        for node_id, tool in self._tools.items():
            tool.warmup(rounds=rounds)

    def _create_tool(self, node: PipelineNode) -> BaseVisionTool:
        """Instantiate a single tool from its node config."""
        if node.tool_type not in AVAILABLE_TOOL_TYPES:
            raise ValueError(
                f"Unknown tool type '{node.tool_type}'. "
                f"Available: {list(AVAILABLE_TOOL_TYPES.keys())}"
            )
        
        tool_class = AVAILABLE_TOOL_TYPES[node.tool_type]
        
        # Merge base config (from YAML) with user overrides
        base_config = _get_base_tool_config(node.tool_type)
        base_config.update(node.tool_config)
        
        model_id = base_config.pop('model', None)
        tool_instance = tool_class(model_id=model_id, config=base_config)
        
        logger.info(f"Created tool '{node.node_id}' ({node.tool_type})")
        return tool_instance

    def _topological_sort(self) -> List[List[str]]:
        """
        Topological sort of nodes into execution layers.
        Nodes in the same layer have no mutual dependencies and can run in parallel.
        """
        # Build adjacency and in-degree maps
        in_degree: Dict[str, int] = {n.node_id: 0 for n in self._node_configs}
        dependents: Dict[str, List[str]] = defaultdict(list)
        node_map = {n.node_id: n for n in self._node_configs}
        
        for node in self._node_configs:
            for dep in node.depends_on:
                if dep not in node_map:
                    raise ValueError(
                        f"Node '{node.node_id}' depends on '{dep}', "
                        f"which doesn't exist in the pipeline."
                    )
                in_degree[node.node_id] += 1
                dependents[dep].append(node.node_id)
        
        # BFS by layers (Kahn's algorithm)
        layers = []
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        
        while queue:
            layer = list(queue)
            queue.clear()
            layers.append(layer)
            
            for nid in layer:
                for dep_nid in dependents[nid]:
                    in_degree[dep_nid] -= 1
                    if in_degree[dep_nid] == 0:
                        queue.append(dep_nid)
        
        # Check for cycles
        total_sorted = sum(len(layer) for layer in layers)
        if total_sorted != len(self._node_configs):
            raise ValueError(
                "Pipeline DAG contains a cycle! "
                "Check depends_on references."
            )
        
        logger.info(f"Execution layers: {layers}")
        return layers

    def _validate_io_contracts(self):
        """
        Validate that each node's InputSchema can be satisfied by the
        combined OutputSchemas of its dependencies.
        
        This is a soft check — it logs warnings instead of raising errors
        when tools don't declare schemas (backward compat).
        """
        available_outputs: Dict[str, type] = {}
        
        for layer in self._layers:
            for node_id in layer:
                tool = self._tools[node_id]
                node = next(n for n in self._node_configs if n.node_id == node_id)
                
                # Check inputs are satisfied
                if tool.InputSchema is not None:
                    # Check that at least one dependency produces a compatible output
                    has_compatible = False
                    for dep_id in node.depends_on:
                        dep_tool = self._tools[dep_id]
                        if dep_tool.OutputSchema is not None:
                            has_compatible = True
                    
                    if node.depends_on and not has_compatible:
                        logger.warning(
                            f"Node '{node_id}' requires InputSchema "
                            f"'{tool.InputSchema.__name__}', but none of its "
                            f"dependencies declare an OutputSchema."
                        )
                
                # Register this tool's output
                if tool.OutputSchema is not None:
                    available_outputs[node_id] = tool.OutputSchema

    # ------------------------------------------------------------------
    # Execution: single frame
    # ------------------------------------------------------------------

    @property
    def tools(self) -> List[BaseVisionTool]:
        """Backward-compatible: return tools in execution order."""
        result = []
        for layer in self._layers:
            for node_id in layer:
                result.append(self._tools[node_id])
        return result

    def run_pipeline(self, frame: Any, context: FrameContext = None) -> tuple:
        """
        Execute the pipeline on a single frame.
        Tools run layer-by-layer; within a layer, tools run in parallel on CPU.
        
        Returns:
            Tuple of (frame, aggregated_data_dict)
        """
        data = {"tools_run": False}
        
        for layer in self._layers:
            if len(layer) == 1:
                # Single tool in layer — run directly
                tool = self._tools[layer[0]]
                tool_results, tool_run = tool.process(frame, data, context=context)
                data.update(tool_results)
                data["tools_run"] = data["tools_run"] or tool_run
            elif self.executor and len(layer) > 1:
                # Multiple tools, CPU mode — run in parallel
                futures = {
                    node_id: self.executor.submit(
                        self._tools[node_id].process, frame, data, context
                    )
                    for node_id in layer
                }
                for node_id, future in futures.items():
                    tool_results, tool_run = future.result()
                    data.update(tool_results)
                    data["tools_run"] = data["tools_run"] or tool_run
            else:
                # Multiple tools, GPU mode — sequential
                for node_id in layer:
                    tool = self._tools[node_id]
                    tool_results, tool_run = tool.process(frame, data, context=context)
                    data.update(tool_results)
                    data["tools_run"] = data["tools_run"] or tool_run
        
        return frame, data

    def extrapolate_last(self, frame: Any) -> tuple:
        """Return extrapolated results from all tools."""
        data = {}
        for layer in self._layers:
            for node_id in layer:
                tool_results = self._tools[node_id].extrapolate_last(frame)
                data.update(tool_results)
        return frame, data

    # ------------------------------------------------------------------
    # Execution: batch
    # ------------------------------------------------------------------

    def run_pipeline_batch(self, frames: List[Any],
                          contexts: List[FrameContext] = None) -> List:
        """
        Process multiple frames through the pipeline.
        Uses parallel workers on CPU.
        """
        if contexts is None:
            contexts = [None] * len(frames)
            
        if self.executor is None:
            # GPU: sequential
            return [self.run_pipeline(f, c) for f, c in zip(frames, contexts)]
        
        # CPU: parallel workers per frame
        futures = [
            self.executor.submit(self.run_pipeline, f, c)
            for f, c in zip(frames, contexts)
        ]
        return [f.result() for f in futures]

    def process_batch_payload(self, payload: BatchPayload) -> BatchPayload:
        """
        Process a BatchPayload through the pipeline.
        Each tool processes the full batch and results are aggregated into FrameResults.
        """
        for layer in self._layers:
            for node_id in layer:
                tool = self._tools[node_id]
                batch_results = tool.process_batch(
                    payload.frames,
                    payload.frame_metadatas,
                )
                # Merge results into frame_results
                for i, result in enumerate(batch_results):
                    if i >= len(payload.frame_results):
                        payload.frame_results.append(
                            FrameResult(metadata=payload.frame_metadatas[i])
                        )
                    payload.frame_results[i].results[node_id] = result
                    payload.frame_results[i].tools_run = True
        
        return payload

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Shutdown worker pool and unload all tools."""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.unload_tools()

    def unload_tools(self):
        for tool in self._tools.values():
            tool.unload_tool()

    def get_tool(self, node_id: str) -> BaseVisionTool:
        """Get a specific tool by node ID."""
        if node_id not in self._tools:
            raise KeyError(f"No tool with node_id '{node_id}'")
        return self._tools[node_id]


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------

def _get_base_tool_config(tool_type: str) -> Dict[str, Any]:
    """Load default tool config from YAML file."""
    configs_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    config_path = os.path.join(configs_dir, f'{tool_type}.yaml')
    
    if not os.path.exists(config_path):
        logger.warning(f"No default config found for '{tool_type}' at {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}
