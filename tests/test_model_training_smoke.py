from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from vision_tools.backends.classification.yolo_cls import YoloClassificationBackend
from vision_tools.backends.detection.yolo_detector import YoloDetectorBackend
from vision_tools.backends.segmentation.yolo_seg import YoloSegmentationBackend
from vision_tools.core.node import NodeState
from vision_tools.nodes.model.classification import Classifier
from vision_tools.nodes.model.detection import Detector
from vision_tools.nodes.model.segmentation import Segmenter
from vision_tools.training.annotations import Annotation, AnnotatedFrame
from vision_tools.training.artifacts import TrainedArtifact
from vision_tools.training.dataset import VisionDataset


def _create_image(path: Path) -> None:
    import cv2

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _dataset(tmp_path: Path, entries: list[tuple[str, AnnotatedFrame]], name: str) -> VisionDataset:
    return VisionDataset.from_entries(
        entries,
        output_dir=str(tmp_path / name),
        val_split=0.0,
    )


class _StubDetectorTrainingBackend(YoloDetectorBackend):
    def train(self, model_ref: str, dataset, config: dict | None = None, callbacks=None) -> TrainedArtifact:
        _ = (callbacks,)
        return TrainedArtifact(
            artifact_path="/tmp/detector-best.pt",
            backend_task="detection",
            backend_model="yolo_detector",
            task="detection",
            model_family="yolo_detector",
            base_checkpoint_id=model_ref,
            metadata={"data_ref": dataset.materialize_for_task("detection"), "config": dict(config or {})},
        )


class _StubClassifierTrainingBackend(YoloClassificationBackend):
    def train(self, model_ref: str, dataset, config: dict | None = None, callbacks=None) -> TrainedArtifact:
        _ = (callbacks,)
        return TrainedArtifact(
            artifact_path="/tmp/classifier-best.pt",
            backend_task="classification",
            backend_model="yolo_cls",
            task="classification",
            model_family="yolo_cls",
            base_checkpoint_id=model_ref,
            metadata={"data_ref": dataset.materialize_for_task("classification"), "config": dict(config or {})},
        )


class _StubSegmentationTrainingBackend(YoloSegmentationBackend):
    def train(self, model_ref: str, dataset, config: dict | None = None, callbacks=None) -> TrainedArtifact:
        _ = (callbacks,)
        return TrainedArtifact(
            artifact_path="/tmp/segmenter-best.pt",
            backend_task="segmentation",
            backend_model="yolo_seg",
            task="segmentation",
            model_family="yolo_seg",
            base_checkpoint_id=model_ref,
            metadata={"data_ref": dataset.materialize_for_task("segmentation"), "config": dict(config or {})},
        )


def test_detector_train_smoke_returns_artifact_and_restores_ready_state(tmp_path: Path) -> None:
    image_path = tmp_path / "detector.jpg"
    _create_image(image_path)
    dataset = _dataset(
        tmp_path,
        [
            (
                str(image_path),
                AnnotatedFrame(annotations=[Annotation.from_bbox(0.1, 0.1, 0.4, 0.4, class_name="box")]),
            )
        ],
        "detector-dataset",
    )
    node = Detector(
        config={"artifact_path": "/tmp/base-detector.pt"},
        backend=_StubDetectorTrainingBackend(),
    )

    artifact = asyncio.run(node.train(dataset, {"epochs": 1}))

    assert artifact.backend_model == "yolo_detector"
    assert node.state == NodeState.READY


def test_detector_train_rejects_non_detection_dataset(tmp_path: Path) -> None:
    image_path = tmp_path / "detector.jpg"
    _create_image(image_path)
    dataset = _dataset(
        tmp_path,
        [(str(image_path), AnnotatedFrame(image_label="cat"))],
        "detector-invalid",
    )
    node = Detector(
        config={"artifact_path": "/tmp/base-detector.pt"},
        backend=_StubDetectorTrainingBackend(),
    )

    with pytest.raises(ValueError, match="object annotations"):
        asyncio.run(node.train(dataset, {"epochs": 1}))

    assert node.state == NodeState.FAILED


def test_classifier_train_smoke_uses_image_label(tmp_path: Path) -> None:
    entries: list[tuple[str, AnnotatedFrame]] = []
    for index, label in enumerate(("cat", "dog")):
        image_path = tmp_path / f"classifier-{index}.jpg"
        _create_image(image_path)
        entries.append((str(image_path), AnnotatedFrame(image_label=label)))
    dataset = _dataset(tmp_path, entries, "classifier-dataset")
    node = Classifier(
        config={"artifact_path": "/tmp/base-classifier.pt"},
        backend=_StubClassifierTrainingBackend(),
    )

    artifact = asyncio.run(node.train(dataset, {"epochs": 1}))

    assert artifact.backend_model == "yolo_cls"
    assert node.state == NodeState.READY


def test_classifier_train_rejects_missing_image_label(tmp_path: Path) -> None:
    entries: list[tuple[str, AnnotatedFrame]] = []
    for index in range(2):
        image_path = tmp_path / f"classifier-missing-{index}.jpg"
        _create_image(image_path)
        entries.append(
            (str(image_path), AnnotatedFrame(image_label="cat" if index == 0 else None))
        )
    dataset = _dataset(tmp_path, entries, "classifier-invalid")
    node = Classifier(
        config={"artifact_path": "/tmp/base-classifier.pt"},
        backend=_StubClassifierTrainingBackend(),
    )

    with pytest.raises(ValueError, match="image_label"):
        asyncio.run(node.train(dataset, {"epochs": 1}))

    assert node.state == NodeState.FAILED


def test_segmenter_train_smoke_returns_artifact(tmp_path: Path) -> None:
    image_path = tmp_path / "segmenter.jpg"
    _create_image(image_path)
    dataset = _dataset(
        tmp_path,
        [
            (
                str(image_path),
                AnnotatedFrame(
                    annotations=[
                        Annotation(
                            class_name="road",
                            polygon=[(0.1, 0.1), (0.8, 0.1), (0.5, 0.9)],
                        )
                    ]
                ),
            )
        ],
        "segmenter-dataset",
    )
    node = Segmenter(
        config={"artifact_path": "/tmp/base-segmenter.pt"},
        backend=_StubSegmentationTrainingBackend(),
    )

    artifact = asyncio.run(node.train(dataset, {"epochs": 1}))

    assert artifact.backend_model == "yolo_seg"
    assert node.state == NodeState.READY


def test_segmenter_train_rejects_dataset_without_annotations(tmp_path: Path) -> None:
    image_path = tmp_path / "segmenter.jpg"
    _create_image(image_path)
    dataset = _dataset(
        tmp_path,
        [(str(image_path), AnnotatedFrame(image_label="scene"))],
        "segmenter-invalid",
    )
    node = Segmenter(
        config={"artifact_path": "/tmp/base-segmenter.pt"},
        backend=_StubSegmentationTrainingBackend(),
    )

    with pytest.raises(ValueError, match="polygon annotations"):
        asyncio.run(node.train(dataset, {"epochs": 1}))

    assert node.state == NodeState.FAILED
