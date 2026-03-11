from __future__ import annotations

from pathlib import Path

import pytest

from vision_tools.capabilities import list_capabilities
from vision_tools.core.node import NodeState
from vision_tools.core.registry import NodeRegistry
import vision_tools.nodes as nodes_pkg
from vision_tools.nodes.model.model_node import ModelNode
from vision_tools.training.annotations import AnnotatedFrame
from vision_tools.training.artifacts import TrainedArtifact
from vision_tools.training.dataset import VisionDataset


class _StubBackend:
    def __init__(self) -> None:
        self.trained_with: dict | None = None

    def load_model(self, model_path: str, device: str = "auto"):
        return {"path": model_path, "device": device}

    def infer(self, model, inputs):
        return {"ok": True}

    def postprocess(self, raw_output, context):
        _ = (raw_output, context)
        return {"result": True}

    def get_available_runtimes(self) -> list[str]:
        return ["cpu"]

    def supports_training(self, config=None) -> bool:
        return True

    def validate_training_dataset(self, dataset, config=None) -> list[str]:
        _ = config
        errors = []
        if dataset.num_images == 0:
            errors.append("empty")
        try:
            dataset.materialize_for_task("classification")
        except Exception as exc:
            errors.append(str(exc))
        return errors

    def train(self, model_ref, dataset, config=None, callbacks=None) -> TrainedArtifact:
        _ = callbacks
        self.trained_with = {
            "model_ref": model_ref,
            "dataset_ref": dataset.materialize_for_task("classification"),
            "config": dict(config or {}),
        }
        return TrainedArtifact(
            artifact_path="/tmp/fake.pt",
            backend_task="classification",
            backend_model="stub",
            task="classification",
            model_family="stub",
            base_checkpoint_id=model_ref,
        )


class _StubNode(ModelNode):
    OutputPorts = {"result": "bool"}

    def preprocess(self, inputs, context):
        _ = context
        return inputs

    def normalize_outputs(self, outputs):
        return outputs

    @classmethod
    def get_training_metadata(cls):
        return {"supported": True, "task": "classification", "annotation_type": "classification"}


def test_model_node_train_delegates_to_backend(tmp_path: Path) -> None:
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake")
    dataset = VisionDataset.from_entries(
        [(str(image_path), AnnotatedFrame(image_label="cat"))],
        output_dir=str(tmp_path / "dataset"),
        val_split=0.0,
    )
    backend = _StubBackend()
    node = _StubNode(
        node_id="trainer",
        config={"checkpoint_id": "base.pt", "artifact_path": None},
        backend=backend,
    )

    import asyncio

    artifact = asyncio.run(node.train(dataset, {"epochs": 1}))

    assert artifact.artifact_path == "/tmp/fake.pt"
    assert backend.trained_with is not None
    assert backend.trained_with["model_ref"] == "base.pt"
    assert backend.trained_with["dataset_ref"].endswith("classification")
    assert node.state == NodeState.READY


def test_capabilities_expose_training_metadata_for_new_nodes() -> None:
    capabilities = list_capabilities()
    nodes = {node["type"]: node for node in capabilities["nodes"]}

    assert nodes["open_vocab_detector"]["training"]["supported"] is False
    assert nodes["detector"]["training"]["supported"] is True
    assert nodes["classifier"]["training"]["annotation_type"] == "classification"
    assert nodes["segmenter"]["training"]["annotation_type"] == "segmentation"


def test_dataset_materialize_for_task_is_cached(tmp_path: Path) -> None:
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake")
    dataset = VisionDataset.from_entries(
        [(str(image_path), AnnotatedFrame(image_label="cat"))],
        output_dir=str(tmp_path / "dataset"),
        val_split=0.0,
    )

    first = dataset.materialize_for_task("classification")
    second = dataset.materialize_for_task("classification")

    assert first == second
    assert Path(first).exists()


def test_model_node_train_fails_fast_for_incompatible_dataset(tmp_path: Path) -> None:
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"fake")
    dataset = VisionDataset.from_entries(
        [(str(image_path), AnnotatedFrame())],
        output_dir=str(tmp_path / "dataset"),
        val_split=0.0,
    )
    backend = _StubBackend()
    node = _StubNode(
        node_id="trainer",
        config={"checkpoint_id": "base.pt", "artifact_path": None},
        backend=backend,
    )

    import asyncio

    with pytest.raises(ValueError, match="dataset is incompatible"):
        asyncio.run(node.train(dataset, {"epochs": 1}))

    assert node.state == NodeState.FAILED
