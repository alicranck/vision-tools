"""
vision-tools — Modular computer vision pipeline library.

Public API
==========

Core:
    Node, NodeState, NodeContext — Base node abstractions
    NodeRegistry — Registry for node discovery and creation
    NodeConfig, PipelineConfig, ExecutionConfig — Configuration models

Nodes:
    ModelNode — Base for ML model-backed nodes
    LogicNode — Base for custom processing logic
    RemoteNode — REST proxy for distributed execution
    ObjectDetector, Embedder, Captioner, PoseEstimator — Task nodes

Pipeline:
    Pipeline — High-level pipeline facade (build, load, run, shutdown)
    DAG, PipelineExecutor, SchemaValidator — Pipeline internals

Backends:
    Backend — Protocol for model backends
    BackendRegistry — Registry for model backends

Runtime:
    ModelResolver, ModelCache — Model management services
    StateManager — Temporal state tracking and rule evaluation

Schemas:
    BoundingBox, DetectionResult, Embedding, EmbeddingResult,
    Caption, CaptionResult, PoseResult, PoseKeypoints, Keypoint,
    FrameMetadata, FrameResult, BatchPayload

Engine:
    VideoInferenceEngine — Video processing orchestration
"""

# Core abstractions
from vision_tools.core.node import Node, NodeState, NodeContext
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.config import (
    NodeConfig,
    PipelineConfig,
    ExecutionConfig,
    ModelIntent,
    ModelSize,
    DeviceTarget,
    InferenceTask,
)

# Pipeline
from vision_tools.pipeline import Pipeline
from vision_tools.pipeline.graph import DAG
from vision_tools.pipeline.executor import PipelineExecutor
from vision_tools.pipeline.validator import SchemaValidator

# Node types
from vision_tools.nodes.model_node import ModelNode
from vision_tools.nodes.logic_node import LogicNode
from vision_tools.nodes.remote_node import RemoteNode
from vision_tools.nodes.aggregator_node import AggregatorNode
from vision_tools.nodes.dynamic_logic_node import DynamicLogicNode

# Backends
from vision_tools.backends.base import Backend
from vision_tools.backends.registry import BackendRegistry

# Runtime services
from vision_tools.runtime.model_resolver import ModelResolver
from vision_tools.runtime.model_cache import ModelCache
from vision_tools.runtime.model_catalog import ModelCatalog, ResolvedModelSpec
from vision_tools.capabilities import list_capabilities

# Schemas (most commonly needed)
from vision_tools.core.schemas import (
    BoundingBox, DetectionResult,
    Embedding, EmbeddingResult,
    Caption, CaptionResult,
    PoseResult, PoseKeypoints, Keypoint,
    FrameMetadata, FrameResult, BatchPayload,
)

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("vision-tools")
except Exception:
    __version__ = "0.1.0"
