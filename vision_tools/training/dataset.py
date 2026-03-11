"""
VisionDataset — Lazy-loading dataset abstraction for training vision tools.

Supports:
- COCO JSON format (primary)
- YOLO format (directory with images/ and labels/)
- Framework-native datasets (Ultralytics, HuggingFace)

The dataset wraps the underlying storage and provides a unified interface
for the ToolTrainer, while deferring heavy I/O to the training framework.
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tempfile
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .annotations import AnnotatedFrame

logger = logging.getLogger(__name__)


class DatasetFormat(str, Enum):
    """Supported dataset formats."""
    COCO = "coco"       # COCO JSON annotation format
    YOLO = "yolo"       # Ultralytics YOLO directory format (images/ + labels/)
    CUSTOM = "custom"   # User-defined format with annotation loader


class DatasetSplit(BaseModel):
    """Metadata for a single dataset split (train/val/test)."""
    images_dir: str = Field(..., description="Path to image directory")
    annotations: Optional[str] = Field(
        None, description="Path to annotation file (COCO JSON) or labels dir (YOLO)"
    )
    num_images: int = Field(0, ge=0)


class VisionDataset:
    """
    Lazy-loading dataset abstraction for training vision tools.
    
    Does NOT load images into memory — instead provides metadata and
    paths that the training framework uses directly.
    
    Args:
        path: Path to dataset root or config file (YOLO .yaml, COCO .json)
        format: Dataset format (auto-detected if not specified)
        name: Human-readable dataset name
    """

    def __init__(self, path: str, format: Optional[DatasetFormat] = None,
                 name: Optional[str] = None):
        self.path = Path(path)
        self.name = name or self.path.stem
        self.format = format or self._detect_format()
        self._metadata: Dict[str, Any] = {}
        self._splits: Dict[str, DatasetSplit] = {}
        self._task_refs: Dict[str, str] = {}
        self._entries: List[tuple[str, "AnnotatedFrame"]] | None = None
        
        # Lazy load metadata on first access
        self._loaded = False

    def _detect_format(self) -> DatasetFormat:
        """Auto-detect dataset format from path."""
        if self.path.suffix == '.json':
            return DatasetFormat.COCO
        elif self.path.suffix in ('.yaml', '.yml'):
            return DatasetFormat.YOLO
        elif self.path.is_dir():
            # Check for YOLO structure
            if (self.path / 'images').exists() and (self.path / 'labels').exists():
                return DatasetFormat.YOLO
            # Check for COCO structure
            if (self.path / 'annotations').exists():
                return DatasetFormat.COCO
        return DatasetFormat.CUSTOM

    def _lazy_load(self):
        """Load dataset metadata on first access."""
        if self._loaded:
            return
        
        if self.format == DatasetFormat.COCO:
            self._load_coco_metadata()
        elif self.format == DatasetFormat.YOLO:
            self._load_yolo_metadata()
        
        self._loaded = True

    def _load_coco_metadata(self):
        """Load COCO JSON metadata (categories, image count) without loading images."""
        json_path = self.path if self.path.suffix == '.json' else self.path / 'annotations' / 'instances.json'
        
        if not json_path.exists():
            logger.warning(f"COCO annotations not found at {json_path}")
            return
        
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        self._metadata = {
            'categories': {c['id']: c['name'] for c in coco_data.get('categories', [])},
            'num_images': len(coco_data.get('images', [])),
            'num_annotations': len(coco_data.get('annotations', [])),
        }
        
        images_dir = str(json_path.parent.parent / 'images') if json_path.parent.name == 'annotations' else str(json_path.parent)
        self._splits['train'] = DatasetSplit(
            images_dir=images_dir,
            annotations=str(json_path),
            num_images=self._metadata['num_images'],
        )
        logger.info(f"COCO dataset: {self._metadata['num_images']} images, "
                     f"{len(self._metadata['categories'])} categories")

    def _load_yolo_metadata(self):
        """Load YOLO directory metadata."""
        if self.path.suffix in ('.yaml', '.yml'):
            import yaml
            with open(self.path, 'r') as f:
                yolo_config = yaml.safe_load(f)
            
            self._metadata = {
                'classes': yolo_config.get('names', {}),
                'num_classes': yolo_config.get('nc', 0),
                'yaml_path': str(self.path),
            }
            
            # Count images if paths are available
            root = self.path.parent
            for split_name in ['train', 'val', 'test']:
                split_path = yolo_config.get(split_name)
                if split_path:
                    full_path = root / split_path if not Path(split_path).is_absolute() else Path(split_path)
                    num_images = len(list(full_path.glob('*.*'))) if full_path.exists() else 0
                    self._splits[split_name] = DatasetSplit(
                        images_dir=str(full_path),
                        num_images=num_images,
                    )
        else:
            # Directory-based YOLO format
            images_dir = self.path / 'images'
            labels_dir = self.path / 'labels'
            if images_dir.exists():
                num_images = len(list(images_dir.glob('*.*')))
                self._splits['train'] = DatasetSplit(
                    images_dir=str(images_dir),
                    annotations=str(labels_dir) if labels_dir.exists() else None,
                    num_images=num_images,
                )
                self._metadata['num_images'] = num_images

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> Dict[str, Any]:
        self._lazy_load()
        return self._metadata

    @property
    def splits(self) -> Dict[str, DatasetSplit]:
        self._lazy_load()
        return self._splits

    @property
    def num_images(self) -> int:
        self._lazy_load()
        return sum(s.num_images for s in self._splits.values())

    @property
    def categories(self) -> Dict:
        """Category mapping (ID → name)."""
        self._lazy_load()
        return self._metadata.get('categories', self._metadata.get('classes', {}))

    def get_ref(self) -> str:
        """
        Return the dataset reference string suitable for passing to tool.train().
        For YOLO: returns the yaml path. For COCO: returns the json path.
        """
        if self.format == DatasetFormat.YOLO:
            return self._metadata.get('yaml_path', str(self.path))
        return str(self.path)

    def materialize_for_task(self, task: str) -> str:
        """Return a framework-ready dataset ref for the requested task.

        Path-backed datasets are returned directly. Entry-backed datasets are
        materialized lazily inside the dataset root so each backend can request
        the representation it needs without app-side export branching.
        """
        normalized = task.strip().lower()
        cached = self._task_refs.get(normalized)
        if cached:
            return cached

        if self._entries is None:
            ref = self.get_ref()
            self._task_refs[normalized] = ref
            return ref

        if normalized in {"detection", "open_vocab_detection"}:
            ref = self._materialize_yolo_detection_dataset(segmentation=False)
        elif normalized == "segmentation":
            ref = self._materialize_yolo_detection_dataset(segmentation=True)
        elif normalized == "classification":
            ref = self._materialize_yolo_classification_dataset()
        else:
            raise ValueError(f"Unsupported task materialization request: {task!r}")

        self._task_refs[normalized] = ref
        return ref

    def validate_for_tool(self, tool_type: str) -> List[str]:
        """
        Check if this dataset is compatible with a given tool type.
        Returns a list of validation errors (empty = valid).
        """
        self._lazy_load()
        errors = []
        
        if self.num_images == 0:
            errors.append("Dataset contains no images")
        
        detection_like_tools = {
            "ov_detection",
            "open_vocab_detector",
            "detector",
            "pose_estimation",
            "pose_estimator",
            "segmenter",
        }
        if tool_type in detection_like_tools:
            if self.format not in (DatasetFormat.YOLO, DatasetFormat.COCO) and self._entries is None:
                errors.append(f"Detection/pose tools require YOLO or COCO format, got {self.format}")
        
        return errors

    # ------------------------------------------------------------------
    # Factory: build from in-memory entries (DB → VisionDataset)
    # ------------------------------------------------------------------

    @classmethod
    def from_entries(
        cls,
        entries: List[tuple[str, "AnnotatedFrame"]],
        name: str = "dataset",
        val_split: float = 0.2,
        output_dir: Optional[str] = None,
    ) -> "VisionDataset":
        """Build a VisionDataset from in-memory annotated entries.

        Writes YOLO-format files to *output_dir* (or a temp directory) so
        that existing training backends (Ultralytics etc.) work unchanged.

        Args:
            entries: List of (image_path, annotated_frame) tuples.
            name: Human-readable dataset name.
            val_split: Fraction of entries to use as validation set.
            output_dir: Where to write the YOLO dataset.  If ``None`` a
                temporary directory is created (caller owns cleanup).

        Returns:
            A VisionDataset pointing at the generated ``data.yaml``.
        """
        from .annotations import AnnotatedFrame  # noqa: F811

        if not entries:
            raise ValueError("Cannot build VisionDataset from empty entries")

        # Resolve output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix=f"vt_dataset_{name}_")
        root = Path(output_dir)
        dataset = cls(path=str(root), format=DatasetFormat.CUSTOM, name=name)
        dataset._entries = list(entries)

        class_names = sorted(
            {ann.class_name for _, af in entries for ann in af.annotations}
            | {af.image_label for _, af in entries if af.image_label}
            | {label for _, af in entries for label in af.labels}
        )
        dataset._metadata = {
            "num_images": len(entries),
            "classes": class_names,
            "output_dir": str(root),
        }
        dataset._splits["train"] = DatasetSplit(images_dir=str(root), num_images=len(entries))
        dataset._loaded = True
        dataset._task_refs = {}
        return dataset

    def _ensure_entries(self) -> List[tuple[str, "AnnotatedFrame"]]:
        if self._entries is None:
            raise ValueError("This dataset is not entry-backed.")
        return self._entries

    def annotation_count(self) -> int | None:
        if self._entries is None:
            return None
        return sum(len(frame.annotations) for _, frame in self._entries)

    def polygon_annotation_count(self) -> int | None:
        if self._entries is None:
            return None
        return sum(
            1
            for _, frame in self._entries
            for ann in frame.annotations
            if len(ann.polygon) >= 3
        )

    def classification_validation_errors(self) -> list[str]:
        if self._entries is None:
            return []

        errors: list[str] = []
        labeled_entries = 0
        for index, (_, frame) in enumerate(self._entries):
            if frame.image_label is not None and not frame.image_label.strip():
                errors.append(f"Entry {index} has an empty image_label.")
                continue
            if frame.image_label is not None:
                labeled_entries += 1

        if labeled_entries == 0:
            errors.append("Classification training requires image_label on each entry.")
        if labeled_entries != len(self._entries):
            errors.append("Classification training requires every entry to define image_label.")
        return errors

    def _split_entries(self) -> tuple[list[int], set[int]]:
        entries = self._ensure_entries()
        indices = list(range(len(entries)))
        random.shuffle(indices)
        val_count = max(1, int(len(entries) * 0.2)) if len(entries) > 1 else 0
        return indices, set(indices[:val_count])

    def _materialize_yolo_detection_dataset(self, segmentation: bool) -> str:
        entries = self._ensure_entries()
        root = self.path / ("segmentation" if segmentation else "detection")
        indices, val_indices = self._split_entries()
        _ = indices

        class_names = sorted({ann.class_name for _, af in entries for ann in af.annotations})
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for idx, (image_path, annotated_frame) in enumerate(entries):
            split = "val" if idx in val_indices else "train"
            src = Path(image_path)
            ext = src.suffix or ".jpg"
            dst_name = f"{idx:06d}{ext}"
            dst_img = root / "images" / split / dst_name
            if src.exists():
                try:
                    if dst_img.exists() or dst_img.is_symlink():
                        dst_img.unlink()
                    dst_img.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst_img)
            else:
                logger.warning("Image not found, skipping: %s", image_path)
                continue

            label_path = root / "labels" / split / f"{idx:06d}.txt"
            lines: list[str] = []
            for ann in annotated_frame.annotations:
                cid = class_to_id[ann.class_name]
                if segmentation:
                    lines.append(ann.to_yolo_segment_line(cid))
                elif ann.is_rectangle:
                    lines.append(ann.to_yolo_bbox_line(cid))
                else:
                    lines.append(ann.to_yolo_segment_line(cid))
            label_path.write_text("\n".join(lines) + "\n" if lines else "")

        yaml_path = root / "data.yaml"
        import yaml

        yaml_path.write_text(
            yaml.dump(
                {
                    "path": str(root),
                    "train": "images/train",
                    "val": "images/val",
                    "nc": len(class_names),
                    "names": class_names,
                },
                default_flow_style=False,
            )
        )
        return str(yaml_path)

    def _materialize_yolo_classification_dataset(self) -> str:
        entries = self._ensure_entries()
        root = self.path / "classification"
        indices, val_indices = self._split_entries()
        _ = indices

        errors = self.classification_validation_errors()
        if errors:
            raise ValueError("\n".join(errors))

        labels = sorted({frame.image_label for _, frame in entries if frame.image_label})
        if not labels:
            raise ValueError("Classification training requires image_label.")

        for split in ("train", "val"):
            for label in labels:
                (root / split / label).mkdir(parents=True, exist_ok=True)

        for idx, (image_path, frame) in enumerate(entries):
            if frame.image_label is None:
                continue
            label = frame.image_label
            split = "val" if idx in val_indices else "train"
            src = Path(image_path)
            ext = src.suffix or ".jpg"
            dst = root / split / label / f"{idx:06d}{ext}"
            if src.exists():
                try:
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst)

        return str(root)

    def __repr__(self) -> str:
        return (
            f"VisionDataset(name='{self.name}', format={self.format.value}, "
            f"images={self.num_images})"
        )
