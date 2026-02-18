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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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

    def validate_for_tool(self, tool_type: str) -> List[str]:
        """
        Check if this dataset is compatible with a given tool type.
        Returns a list of validation errors (empty = valid).
        """
        self._lazy_load()
        errors = []
        
        if self.num_images == 0:
            errors.append("Dataset contains no images")
        
        if tool_type in ('ov_detection', 'pose_estimation', 'object_detector', 'pose_estimator'):
            if self.format not in (DatasetFormat.YOLO, DatasetFormat.COCO):
                errors.append(f"Detection/pose tools require YOLO or COCO format, got {self.format}")
        
        return errors

    def __repr__(self) -> str:
        return (
            f"VisionDataset(name='{self.name}', format={self.format.value}, "
            f"images={self.num_images})"
        )
