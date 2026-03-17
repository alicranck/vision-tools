import numpy as np

from vision_tools.core.graph_types import Caption, Embedding, Keypoint, PoseKeypoints, Poses
from vision_tools.core.graph_types import BoundingBox, Detections, Image
from vision_tools.core.node import NodeContext, NodeState
from vision_tools.nodes.model.captioning import Captioner
from vision_tools.nodes.model.embedding import Embedder
from vision_tools.nodes.model.model_node import ModelNode
from vision_tools.nodes.model.pose import PoseEstimator


class StubBackend:
    def load_model(self, model_path, device="auto"):
        return {"model_path": model_path, "device": device}

    def infer(self, model, inputs):
        return {"raw": True, "shape": getattr(inputs, "shape", None)}

    def postprocess(self, raw_output, context):
        return {
            "detections": Detections(
                items=[
                    BoundingBox(
                        xyxy=[0, 0, 4, 4],
                        class_id=0,
                        class_name="person",
                        confidence=0.9,
                    )
                ]
            )
        }

    def get_available_runtimes(self):
        return ["mock"]


class StubDetector(ModelNode):
    InputPorts = {"image": "Image"}
    OutputPorts = {"detections": "Detections"}

    def __init__(self, node_id="det", config=None):
        super().__init__(
            node_id=node_id,
            config=config or {"checkpoint_id": "mock"},
            backend=StubBackend(),
        )

    def preprocess(self, inputs, context):
        return inputs["image"].data


def test_model_node_lifecycle_and_process():
    node = StubDetector()
    assert node.state == NodeState.UNLOADED
    node.load()
    assert node.state == NodeState.READY
    image = Image(data=np.zeros((8, 8, 3), dtype=np.uint8), width=8, height=8, channels=3)
    result = node.process({"image": image}, NodeContext(frame_shape=(8, 8, 3)))
    assert len(result["detections"].items) == 1
    node.unload()
    assert node.state == NodeState.UNLOADED


def test_model_node_verify():
    node = StubDetector()
    node.load()
    assert node.verify() is True


class StubEmbeddingBackend:
    def load_model(self, model_path, device="auto"):
        return {}

    def infer(self, model, inputs):
        return {}

    def postprocess(self, raw_output, context):
        return {"embedding": {"vector": [0.1, 0.2], "model_id": "stub", "dimension": 2}}


class StubCaptionBackend:
    def load_model(self, model_path, device="auto"):
        return {}

    def infer(self, model, inputs):
        return {}

    def postprocess(self, raw_output, context):
        return {"caption": {"text": "hello", "model_id": "stub"}}


class StubPoseBackend:
    def load_model(self, model_path, device="auto"):
        return {}

    def infer(self, model, inputs):
        return {}

    def postprocess(self, raw_output, context):
        return {
            "poses": {
                "items": [
                    {
                        "person_id": 1,
                        "keypoints": [{"x": 1.0, "y": 2.0, "confidence": 0.9}],
                    }
                ]
            }
        }


def test_embedder_normalizes_graph_output():
    node = Embedder(config={"model_family": "siglip2"}, backend=StubEmbeddingBackend())
    node.load()
    image = Image(data=np.zeros((8, 8, 3), dtype=np.uint8), width=8, height=8, channels=3)
    result = node.process({"image": image}, NodeContext(frame_shape=(8, 8, 3)))
    assert isinstance(result["embedding"], Embedding)
    assert result["embedding"].dimension == 2


def test_captioner_normalizes_graph_output():
    node = Captioner(config={"model_family": "smolvlm"}, backend=StubCaptionBackend())
    node.load()
    image = Image(data=np.zeros((8, 8, 3), dtype=np.uint8), width=8, height=8, channels=3)
    result = node.process({"image": image}, NodeContext(frame_shape=(8, 8, 3)))
    assert isinstance(result["caption"], Caption)
    assert result["caption"].text == "hello"


def test_pose_estimator_normalizes_graph_output():
    node = PoseEstimator(config={"model_family": "yolo_pose"}, backend=StubPoseBackend())
    node.load()
    image = Image(data=np.zeros((8, 8, 3), dtype=np.uint8), width=8, height=8, channels=3)
    result = node.process({"image": image}, NodeContext(frame_shape=(8, 8, 3)))
    assert isinstance(result["poses"], Poses)
    assert result["poses"].items[0].person_id == 1
