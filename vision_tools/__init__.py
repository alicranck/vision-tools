from vision_tools.utils.locations import setup_cache_env

setup_cache_env()

from vision_tools.backends.base import Backend
from vision_tools.backends.registry import BackendRegistry
from vision_tools.capabilities import list_capabilities
from vision_tools.core.config import (
    DeviceTarget,
    ExecutionConfig,
    InferenceTask,
    ModelIntent,
    ModelSize,
    NodeConfig,
    PipelineConfig,
)
from vision_tools.core.graph_types import (
    Alert,
    Alerts,
    BoundingBox,
    Caption,
    Crop,
    Crops,
    Detections,
    Embedding,
    FrameInfo,
    History,
    Image,
    Keypoint,
    PoseKeypoints,
    Poses,
    Track,
    Tracks,
)
from vision_tools.core.node import Node, NodeContext, NodeState
from vision_tools.core.registry import NodeRegistry
from vision_tools.core.schemas import BatchPayload, FrameMetadata, FrameResult
from vision_tools.core.sources import SOURCE_NODE_ID, get_source_ports
from vision_tools.nodes.logic.dynamic_logic_node import DynamicLogicNode
from vision_tools.nodes.logic.logic_node import LogicNode
from vision_tools.nodes.model.captioning import Captioner
from vision_tools.nodes.model.classification import Classifier
from vision_tools.nodes.model.detection import Detector, ObjectDetector, OpenVocabularyDetector
from vision_tools.nodes.model.embedding import Embedder
from vision_tools.nodes.model.model_node import ModelNode
from vision_tools.nodes.model.pose import PoseEstimator
from vision_tools.nodes.model.segmentation import Segmenter
from vision_tools.nodes.state.buffer_node import BufferNode
from vision_tools.nodes.state.track_node import TrackNode
from vision_tools.nodes.utility.crop_node import CropNode
from vision_tools.nodes.utility.filter_node import FilterNode
from vision_tools.pipeline import Pipeline
from vision_tools.pipeline.executor import PipelineExecutor
from vision_tools.pipeline.graph import DAG
from vision_tools.pipeline.validator import SchemaValidator
from vision_tools.runtime.model_cache import ModelCache
from vision_tools.runtime.model_catalog import ModelCatalog, ResolvedModelSpec
from vision_tools.runtime.model_resolver import ModelResolver

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("vision-tools")
except Exception:
    __version__ = "0.1.0"
