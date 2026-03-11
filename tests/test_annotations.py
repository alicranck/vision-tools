"""Tests for the unified annotation model and VisionDataset.from_entries()."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vision_tools.training.annotations import Annotation, AnnotatedFrame
from vision_tools.training.dataset import VisionDataset


# ---------------------------------------------------------------------------
# Annotation model
# ---------------------------------------------------------------------------

class TestAnnotation:
    def test_bbox_from_polygon(self):
        ann = Annotation(
            class_name="cat",
            polygon=[(0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)],
        )
        x, y, w, h = ann.bbox
        assert pytest.approx(x) == 0.1
        assert pytest.approx(y) == 0.2
        assert pytest.approx(w) == 0.4
        assert pytest.approx(h) == 0.6

    def test_from_bbox_round_trip(self):
        ann = Annotation.from_bbox(0.1, 0.2, 0.4, 0.6, class_name="dog")
        x, y, w, h = ann.bbox
        assert pytest.approx(x) == 0.1
        assert pytest.approx(y) == 0.2
        assert pytest.approx(w) == 0.4
        assert pytest.approx(h) == 0.6
        assert ann.is_rectangle

    def test_is_rectangle_true(self):
        ann = Annotation.from_bbox(0.0, 0.0, 0.5, 0.5, class_name="box")
        assert ann.is_rectangle

    def test_is_rectangle_false_for_triangle(self):
        ann = Annotation(
            class_name="tri",
            polygon=[(0.0, 0.0), (0.5, 0.0), (0.25, 0.5)],
        )
        assert not ann.is_rectangle

    def test_is_rectangle_false_for_non_axis_aligned(self):
        ann = Annotation(
            class_name="diamond",
            polygon=[(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)],
        )
        assert not ann.is_rectangle

    def test_to_mask(self):
        ann = Annotation.from_bbox(0.0, 0.0, 1.0, 1.0, class_name="full")
        mask = ann.to_mask(10, 10)
        assert mask.shape == (10, 10)
        assert mask.dtype == np.uint8
        # Full-image box → most pixels should be 1
        assert mask.sum() > 80

    def test_to_mask_partial(self):
        ann = Annotation.from_bbox(0.0, 0.0, 0.5, 0.5, class_name="quarter")
        mask = ann.to_mask(100, 100)
        # Roughly a quarter of pixels
        assert 2000 < mask.sum() < 3000

    def test_to_yolo_bbox_line(self):
        ann = Annotation.from_bbox(0.1, 0.2, 0.4, 0.6, class_name="cat")
        line = ann.to_yolo_bbox_line(class_id=3)
        parts = line.split()
        assert parts[0] == "3"
        assert pytest.approx(float(parts[1]), abs=1e-4) == 0.3  # cx = 0.1 + 0.4/2
        assert pytest.approx(float(parts[2]), abs=1e-4) == 0.5  # cy = 0.2 + 0.6/2
        assert pytest.approx(float(parts[3]), abs=1e-4) == 0.4  # w
        assert pytest.approx(float(parts[4]), abs=1e-4) == 0.6  # h

    def test_to_yolo_segment_line(self):
        ann = Annotation(
            class_name="shape",
            polygon=[(0.1, 0.2), (0.3, 0.2), (0.2, 0.5)],
        )
        line = ann.to_yolo_segment_line(class_id=0)
        parts = line.split()
        assert parts[0] == "0"
        assert len(parts) == 7  # class_id + 3 pairs

    def test_serialization_round_trip(self):
        ann = Annotation.from_bbox(0.1, 0.2, 0.3, 0.4, class_name="dog", confidence=0.95)
        data = ann.model_dump()
        restored = Annotation.model_validate(data)
        assert restored.class_name == "dog"
        assert restored.confidence == 0.95
        assert restored.bbox == ann.bbox

    def test_json_round_trip(self):
        ann = Annotation.from_bbox(0.1, 0.2, 0.3, 0.4, class_name="cat")
        json_str = ann.model_dump_json()
        restored = Annotation.model_validate_json(json_str)
        assert restored.class_name == ann.class_name
        assert len(restored.polygon) == 4

    def test_polygon_min_length(self):
        with pytest.raises(Exception):
            Annotation(class_name="bad", polygon=[(0.0, 0.0), (1.0, 1.0)])


# ---------------------------------------------------------------------------
# AnnotatedFrame
# ---------------------------------------------------------------------------

class TestAnnotatedFrame:
    def test_class_names(self):
        af = AnnotatedFrame(annotations=[
            Annotation.from_bbox(0, 0, 0.5, 0.5, class_name="cat"),
            Annotation.from_bbox(0.5, 0.5, 0.5, 0.5, class_name="dog"),
            Annotation.from_bbox(0.1, 0.1, 0.2, 0.2, class_name="cat"),
        ])
        assert af.class_names == ["cat", "dog"]

    def test_is_empty(self):
        assert AnnotatedFrame().is_empty
        assert not AnnotatedFrame(annotations=[
            Annotation.from_bbox(0, 0, 1, 1, class_name="x")
        ]).is_empty

    def test_serialization(self):
        af = AnnotatedFrame(annotations=[
            Annotation.from_bbox(0, 0, 0.5, 0.5, class_name="a"),
        ])
        data = json.loads(af.model_dump_json())
        restored = AnnotatedFrame.model_validate(data)
        assert len(restored.annotations) == 1
        assert restored.annotations[0].class_name == "a"


# ---------------------------------------------------------------------------
# VisionDataset.from_entries()
# ---------------------------------------------------------------------------

def _create_test_image(path: Path, size: tuple[int, int] = (100, 100)):
    """Create a minimal JPEG test image."""
    import cv2
    img = np.zeros((*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


class TestVisionDatasetFromEntries:
    def test_basic_creation(self, tmp_path):
        # Create test images
        img_dir = tmp_path / "source_images"
        img_dir.mkdir()
        entries = []
        for i in range(5):
            img_path = img_dir / f"frame_{i}.jpg"
            _create_test_image(img_path)
            af = AnnotatedFrame(annotations=[
                Annotation.from_bbox(0.1, 0.1, 0.3, 0.3, class_name="cat"),
                Annotation.from_bbox(0.5, 0.5, 0.2, 0.2, class_name="dog"),
            ])
            entries.append((str(img_path), af))

        output_dir = str(tmp_path / "yolo_out")
        ds = VisionDataset.from_entries(entries, name="test_ds", output_dir=output_dir)

        assert ds.format.value == "custom"
        assert ds.name == "test_ds"
        assert ds.num_images == 5
        assert len(ds.categories) == 2

    def test_detection_materialization_creates_yolo_directory_structure(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        entries = []
        for i in range(4):
            p = img_dir / f"img_{i}.jpg"
            _create_test_image(p)
            entries.append((str(p), AnnotatedFrame(annotations=[
                Annotation.from_bbox(0.0, 0.0, 0.5, 0.5, class_name="obj"),
            ])))

        out = tmp_path / "yolo"
        ds = VisionDataset.from_entries(entries, output_dir=str(out))
        ref = ds.materialize_for_task("detection")

        # Check directory structure
        assert ref.endswith("data.yaml")
        assert (out / "detection" / "data.yaml").exists()
        assert (out / "detection" / "images" / "train").is_dir()
        assert (out / "detection" / "images" / "val").is_dir()
        assert (out / "detection" / "labels" / "train").is_dir()
        assert (out / "detection" / "labels" / "val").is_dir()

        # Check label format
        label_files = list((out / "detection" / "labels" / "train").glob("*.txt")) + \
                      list((out / "detection" / "labels" / "val").glob("*.txt"))
        assert len(label_files) == 4

        for lf in label_files:
            content = lf.read_text().strip()
            if content:
                parts = content.split()
                assert parts[0] == "0"  # single class → id 0
                assert len(parts) == 5  # bbox: class cx cy w h

    @pytest.mark.parametrize("tool_type", ["detector", "open_vocab_detector"])
    def test_validate_for_tool(self, tmp_path, tool_type):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        entries = []
        for i in range(3):
            p = img_dir / f"img_{i}.jpg"
            _create_test_image(p)
            entries.append((str(p), AnnotatedFrame(annotations=[
                Annotation.from_bbox(0.1, 0.1, 0.3, 0.3, class_name="defect"),
            ])))

        ds = VisionDataset.from_entries(entries, output_dir=str(tmp_path / "out"))
        errors = ds.validate_for_tool(tool_type)
        assert errors == []

    def test_get_ref_returns_yaml(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        p = img_dir / "img.jpg"
        _create_test_image(p)

        entries = [(str(p), AnnotatedFrame(annotations=[
            Annotation.from_bbox(0, 0, 1, 1, class_name="x"),
        ]))]
        ds = VisionDataset.from_entries(entries, output_dir=str(tmp_path / "out"), val_split=0.0)
        ref = ds.materialize_for_task("detection")
        assert ref.endswith("data.yaml")

    def test_empty_entries_raises(self):
        with pytest.raises(ValueError, match="empty"):
            VisionDataset.from_entries([])

    def test_polygon_annotations_use_segment_format(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        p = img_dir / "img.jpg"
        _create_test_image(p)

        triangle = Annotation(
            class_name="shape",
            polygon=[(0.1, 0.1), (0.5, 0.1), (0.3, 0.8)],
        )
        entries = [(str(p), AnnotatedFrame(annotations=[triangle]))]
        ds = VisionDataset.from_entries(entries, output_dir=str(tmp_path / "out"), val_split=0.0)
        ds.materialize_for_task("segmentation")

        # Should use segment format (class_id + vertex pairs)
        label_files = list((tmp_path / "out" / "segmentation" / "labels" / "train").glob("*.txt"))
        assert len(label_files) == 1
        content = label_files[0].read_text().strip()
        parts = content.split()
        assert parts[0] == "0"
        assert len(parts) == 7  # class_id + 3 vertex pairs

    def test_mixed_rect_and_polygon(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        p = img_dir / "img.jpg"
        _create_test_image(p)

        entries = [(str(p), AnnotatedFrame(annotations=[
            Annotation.from_bbox(0.1, 0.1, 0.3, 0.3, class_name="box_obj"),
            Annotation(class_name="poly_obj", polygon=[(0.5, 0.5), (0.8, 0.5), (0.65, 0.9)]),
        ]))]
        ds = VisionDataset.from_entries(entries, output_dir=str(tmp_path / "out"), val_split=0.0)
        ds.materialize_for_task("detection")

        label_files = list((tmp_path / "out" / "detection" / "labels" / "train").glob("*.txt"))
        lines = label_files[0].read_text().strip().split("\n")
        assert len(lines) == 2
        # Rect uses bbox format (5 values), polygon uses segment format (7 values)
        assert len(lines[0].split()) == 5  # bbox
        assert len(lines[1].split()) == 7  # segment

    def test_classification_materialization_creates_split_dirs(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        entries = []
        for i in range(3):
            p = img_dir / f"img_{i}.jpg"
            _create_test_image(p)
            entries.append((str(p), AnnotatedFrame(image_label="cat" if i < 2 else "dog")))

        ds = VisionDataset.from_entries(entries, output_dir=str(tmp_path / "out"), val_split=0.0)
        ref = ds.materialize_for_task("classification")

        assert ref.endswith("classification")
        assert (tmp_path / "out" / "classification" / "train").is_dir()
        assert (tmp_path / "out" / "classification" / "train" / "cat").is_dir()

    def test_annotated_frame_includes_image_label_and_labels(self):
        frame = AnnotatedFrame(image_label="cat", labels=["outdoor"])
        assert frame.class_names == ["cat", "outdoor"]
        assert not frame.is_empty

    def test_classification_validation_errors_require_image_label_for_every_entry(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        entries = []
        for i in range(2):
            image_path = img_dir / f"img_{i}.jpg"
            _create_test_image(image_path)
            frame = AnnotatedFrame(image_label="cat" if i == 0 else None)
            entries.append((str(image_path), frame))

        dataset = VisionDataset.from_entries(
            entries,
            output_dir=str(tmp_path / "out"),
            val_split=0.0,
        )

        assert dataset.classification_validation_errors() == [
            "Classification training requires every entry to define image_label."
        ]

    def test_classification_materialization_rejects_missing_image_label(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        image_path = img_dir / "img.jpg"
        _create_test_image(image_path)

        dataset = VisionDataset.from_entries(
            [(str(image_path), AnnotatedFrame(labels=["tag-only"]))],
            output_dir=str(tmp_path / "out"),
            val_split=0.0,
        )

        with pytest.raises(ValueError, match="image_label"):
            dataset.materialize_for_task("classification")
