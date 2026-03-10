import asyncio

from vision_tools.nodes.model.model_node import ModelNode
from vision_tools.training.config import TrainConfig
from vision_tools.training.dataset import VisionDataset
from vision_tools.training.trainer import ToolTrainer


class _StubBackend:
    def load_model(self, model_path, device="auto"):
        return None

    def infer(self, model, inputs):
        return {}

    def postprocess(self, raw_output, context):
        return {}

    def get_available_runtimes(self):
        return ["mock"]


class _TrainableNode(ModelNode):
    def __init__(self):
        super().__init__(
            node_id="train_node",
            config={"model": "mock-model"},
            backend=_StubBackend(),
        )
        self.train_calls = []

    async def _train_impl(self, dataset_ref: str, config: dict) -> None:
        self.train_calls.append((dataset_ref, dict(config)))


def test_tool_trainer_supports_model_node(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    (images_dir / "frame0.jpg").write_bytes(b"\x00")

    dataset = VisionDataset(str(tmp_path))
    node = _TrainableNode()

    events = []
    trainer = ToolTrainer(node, on_progress=lambda e: events.append(e))
    asyncio.run(trainer.train(dataset, TrainConfig(epochs=2, batch_size=1)))

    assert len(node.train_calls) == 1
    dataset_ref, kwargs = node.train_calls[0]
    assert dataset_ref == str(tmp_path)
    assert kwargs["epochs"] == 2
    assert kwargs["batch"] == 1
    assert kwargs["data"] == str(tmp_path)
    assert events[0]["event"] == "training_started"
    assert events[-1]["event"] == "training_completed"
    assert events[0]["tool"] == "train_node"
