from __future__ import annotations

import logging
from importlib import import_module

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.registry import NodeRegistry

logger = logging.getLogger(__name__)


NODE_SPECS = [
    ("object_detector", "model", "vision_tools.nodes.model.detection", "OpenVocabularyDetector"),
    ("open_vocab_detector", "model", "vision_tools.nodes.model.detection", "OpenVocabularyDetector"),
    ("detector", "model", "vision_tools.nodes.model.detection", "Detector"),
    ("classifier", "model", "vision_tools.nodes.model.classification", "Classifier"),
    ("segmenter", "model", "vision_tools.nodes.model.segmentation", "Segmenter"),
    ("embedder", "model", "vision_tools.nodes.model.embedding", "Embedder"),
    ("captioner", "model", "vision_tools.nodes.model.captioning", "Captioner"),
    ("pose_estimator", "model", "vision_tools.nodes.model.pose", "PoseEstimator"),
    ("dynamic_logic", "logic", "vision_tools.nodes.logic.dynamic_logic_node", "DynamicLogicNode"),
    ("filter", "utility", "vision_tools.nodes.utility.filter_node", "FilterNode"),
    ("track", "state", "vision_tools.nodes.state.track_node", "TrackNode"),
    ("crop", "utility", "vision_tools.nodes.utility.crop_node", "CropNode"),
    ("apply", "utility", "vision_tools.nodes.utility.apply_node", "ApplyNode"),
    ("buffer", "state", "vision_tools.nodes.state.buffer_node", "BufferNode"),
    # Memory persistence nodes
    ("file_writer", "output", "vision_tools.nodes.io.file_writer_node", "FileWriterNode"),
    ("memory_store", "state", "vision_tools.nodes.state.memory_store_node", "MemoryStoreNode"),
    ("dataset_writer", "output", "vision_tools.nodes.io.dataset_writer_node", "DatasetWriterNode"),
]

BACKEND_SPECS = [
    ("detection", "yolo", "vision_tools.backends.detection.yolo", "YoloBackend"),
    ("detection", "yolo_detector", "vision_tools.backends.detection.yolo_detector", "YoloDetectorBackend"),
    ("detection", "rtdetr", "vision_tools.backends.detection.rtdetr", "RTDetrBackend"),
    ("classification", "yolo_cls", "vision_tools.backends.classification.yolo_cls", "YoloClassificationBackend"),
    ("segmentation", "yolo_seg", "vision_tools.backends.segmentation.yolo_seg", "YoloSegmentationBackend"),
    ("embedding", "siglip2", "vision_tools.backends.embedding.siglip2", "SigLIP2Backend"),
    ("embedding", "clip", "vision_tools.backends.embedding.clip", "CLIPBackend"),
    ("captioning", "smolvlm", "vision_tools.backends.captioning.smolvlm", "SmolVLMBackend"),
    ("captioning", "llamacpp", "vision_tools.backends.captioning.llamacpp", "LlamaCppBackend"),
    ("pose", "yolo_pose", "vision_tools.backends.pose.yolo_pose", "YoloPoseBackend"),
]


def register_all() -> None:
    for name, category, module_name, class_name in NODE_SPECS:
        try:
            module = import_module(module_name)
        except ImportError as exc:
            logger.info(
                "Skipping optional node registration for %s (%s): %s",
                name,
                module_name,
                exc,
            )
            continue
        NodeRegistry.register_class(name, getattr(module, class_name), category=category)

    for task, model, module_name, class_name in BACKEND_SPECS:
        try:
            module = import_module(module_name)
        except ImportError as exc:
            logger.info(
                "Skipping optional backend registration for %s:%s: %s",
                task,
                model,
                exc,
            )
            continue
        BackendRegistry.register_class(task, model, getattr(module, class_name))


register_all()
