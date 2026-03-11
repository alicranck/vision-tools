import vision_tools.nodes as nodes_pkg
from vision_tools.core.config import NodeConfig
from vision_tools.core.registry import NodeRegistry
from vision_tools.nodes.logic.dynamic_logic_node import DynamicLogicNode
from vision_tools.nodes.state.buffer_node import BufferNode
from vision_tools.nodes.state.track_node import TrackNode
from vision_tools.nodes.utility.crop_node import CropNode
from vision_tools.nodes.utility.filter_node import FilterNode


def setup_function():
    NodeRegistry.clear()
    nodes_pkg.register_all()


def test_builtin_nodes_registered():
    names = {entry["type"] for entry in NodeRegistry.list_nodes()}
    assert "open_vocab_detector" in names
    assert "detector" in names
    assert "classifier" in names
    assert "segmenter" in names
    assert "embedder" in names
    assert "captioner" in names
    assert "pose_estimator" in names
    assert "filter" in names
    assert "track" in names
    assert "crop" in names
    assert "buffer" in names
    assert "dynamic_logic" in names
    assert "remote" not in names


def test_registry_metadata_exposes_ports():
    metadata = {entry["type"]: entry for entry in NodeRegistry.list_nodes()}
    sources = {entry["type"]: entry for entry in NodeRegistry.list_sources()}
    assert metadata["filter"]["input_ports"]["detections"] == "Detections"
    assert metadata["track"]["output_ports"]["tracks"] == "Tracks"
    assert metadata["embedder"]["output_ports"]["embedding"] == "Embedding"
    assert metadata["captioner"]["output_ports"]["caption"] == "Caption"
    assert metadata["pose_estimator"]["output_ports"]["poses"] == "Poses"
    assert metadata["dynamic_logic"]["dynamic_ports"] is True
    assert metadata["detector"]["training"]["supported"] is True
    assert metadata["open_vocab_detector"]["training"]["supported"] is False
    assert sources["input"]["output_ports"]["image"] == "Image"
    assert sources["input"]["executable"] is False


def test_registry_create_filter():
    node = NodeRegistry.create(
        NodeConfig(
            node_id="filter1",
            node_type="filter",
            config={"min_confidence": 0.5},
            inputs={"detections": "det.detections"},
        )
    )
    assert isinstance(node, FilterNode)
