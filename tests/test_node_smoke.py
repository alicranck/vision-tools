from __future__ import annotations

import numpy as np

from vision_tools.core.graph_types import (
    BoundingBox,
    Caption,
    Classification,
    Detections,
    Embedding,
    History,
    Image,
    Keypoint,
    PoseKeypoints,
    Poses,
    SegmentationMask,
    Track,
    Tracks,
)
from vision_tools.core.node import NodeContext
from vision_tools.nodes.logic.dynamic_logic_node import DynamicLogicNode
from vision_tools.nodes.model.captioning import Captioner
from vision_tools.nodes.model.classification import Classifier
from vision_tools.nodes.model.detection import Detector, OpenVocabularyDetector
from vision_tools.nodes.model.embedding import Embedder
from vision_tools.nodes.model.pose import PoseEstimator
from vision_tools.nodes.model.segmentation import Segmenter
from vision_tools.nodes.state.buffer_node import BufferNode
from vision_tools.nodes.state.track_node import TrackNode
from vision_tools.nodes.utility.crop_node import CropNode
from vision_tools.nodes.utility.filter_node import FilterNode


class _StubInferenceBackend:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.loaded_model_path: str | None = None
        self.last_inputs = None

    def configure(self, config: dict) -> None:
        self.config = dict(config)

    def load_model(self, model_path: str, device: str = "auto"):
        self.loaded_model_path = model_path
        return {"model_path": model_path, "device": device}

    def infer(self, model, inputs):
        _ = model
        self.last_inputs = inputs
        return {"ok": True}

    def postprocess(self, raw_output, context):
        _ = (raw_output, context)
        return self.payload

    def get_available_runtimes(self) -> list[str]:
        return ["cpu"]


def _image() -> Image:
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    return Image(data=pixels, width=16, height=16, channels=3)


def test_model_nodes_load_and_process_with_canonical_outputs() -> None:
    image = _image()
    context = NodeContext(frame_shape=(16, 16, 3))

    cases = [
        (
            OpenVocabularyDetector(
                config={"artifact_path": "/tmp/open-vocab.pt", "model_family": "yolo"},
                backend=_StubInferenceBackend(
                    {"detections": Detections(items=[BoundingBox(xyxy=[0, 0, 5, 5], class_name="person")])}
                ),
            ),
            "detections",
        ),
        (
            Detector(
                config={"artifact_path": "/tmp/detector.pt"},
                backend=_StubInferenceBackend(
                    {"detections": Detections(items=[BoundingBox(xyxy=[1, 1, 6, 6], class_name="car")])}
                ),
            ),
            "detections",
        ),
        (
            Classifier(
                config={"artifact_path": "/tmp/classifier.pt"},
                backend=_StubInferenceBackend(
                    {"classifications": {"items": [{"class_name": "cat", "confidence": 0.9}]}}
                ),
            ),
            "classifications",
        ),
        (
            Segmenter(
                config={"artifact_path": "/tmp/segmenter.pt"},
                backend=_StubInferenceBackend(
                    {
                        "segmentation_masks": {
                            "items": [
                                {
                                    "polygon": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
                                    "class_name": "road",
                                    "confidence": 0.8,
                                }
                            ]
                        }
                    }
                ),
            ),
            "segmentation_masks",
        ),
        (
            Embedder(
                config={"artifact_path": "/tmp/embedder.pt", "model": "siglip2"},
                backend=_StubInferenceBackend(
                    {"embedding": Embedding(vector=[0.1, 0.2], model_id="stub", dimension=2)}
                ),
            ),
            "embedding",
        ),
        (
            Captioner(
                config={"artifact_path": "/tmp/captioner.gguf", "model": "smolvlm"},
                backend=_StubInferenceBackend({"caption": Caption(text="a test caption", model_id="stub")}),
            ),
            "caption",
        ),
        (
            PoseEstimator(
                config={"artifact_path": "/tmp/pose.pt", "model": "yolo_pose"},
                backend=_StubInferenceBackend(
                    {
                        "poses": Poses(
                            items=[
                                PoseKeypoints(
                                    person_id=0,
                                    keypoints=[Keypoint(x=1.0, y=2.0, confidence=0.9)],
                                )
                            ]
                        )
                    }
                ),
            ),
            "poses",
        ),
    ]

    for node, port_name in cases:
        node.load()
        result = node.process({"image": image}, context)
        assert port_name in result
        assert getattr(node.backend, "loaded_model_path", None) == node.config["artifact_path"]


def test_non_model_nodes_smoke_minimal_process_paths() -> None:
    image = _image()
    detections = Detections(
        items=[BoundingBox(xyxy=[1, 1, 6, 6], class_name="person", confidence=0.9)]
    )

    filter_node = FilterNode("filter", {"min_confidence": 0.5})
    filtered = filter_node.process({"detections": detections}, NodeContext())
    assert len(filtered["detections"].items) == 1

    track_node = TrackNode("track", {"iou_threshold": 0.1, "max_age": 2})
    tracked = track_node.process({"detections": detections}, NodeContext())
    assert tracked["tracks"].items[0].track_id == 1

    crop_node = CropNode("crop", {"padding": 0.0})
    cropped = crop_node.process({"image": image, "regions": tracked["tracks"]}, NodeContext())
    assert len(cropped["images"].items) == 1

    buffer_node = BufferNode(
        "buffer",
        {"item_type": "Tracks", "window_size_seconds": 5.0, "emit_every_frames": 1},
    )
    history = buffer_node.process(
        {"items": Tracks(items=[Track(track_id=1, xyxy=[0, 0, 1, 1])])},
        NodeContext(frame_idx=0, timestamp=0.0),
    )
    assert isinstance(history["history"], History)

    dynamic_logic = DynamicLogicNode(
        "logic",
        {
            "input_ports": {"tracks": "Tracks"},
            "output_ports": {"detections": "Detections"},
            "code": """
def execute(inputs, context, config, state):
    return {"detections": {"items": [{"xyxy": [0, 0, 2, 2], "class_name": "person"}]}}
""",
        },
    )
    dynamic_result = dynamic_logic.process({"tracks": tracked["tracks"]}, NodeContext())
    assert len(dynamic_result["detections"].items) == 1
