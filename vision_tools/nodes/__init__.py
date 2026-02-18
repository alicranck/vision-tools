"""
Nodes package — pipeline node implementations.

Includes an idempotent bootstrap helper to (re)register built-in nodes/backends
after tests or callers clear registries.
"""
from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.registry import NodeRegistry

# Import modules so class symbols are available.
from vision_tools.nodes import detection
from vision_tools.nodes import embedding
from vision_tools.nodes import captioning
from vision_tools.nodes import pose
from vision_tools.nodes import remote_node

from vision_tools.backends.detection import yolo
from vision_tools.backends.detection import rtdetr
from vision_tools.backends.embedding import siglip2
from vision_tools.backends.embedding import clip
from vision_tools.backends.captioning import smolvlm
from vision_tools.backends.captioning import llamacpp
from vision_tools.backends.pose import yolo_pose


def register_all() -> None:
    """Idempotently register all built-in nodes and backends."""
    NodeRegistry._registry["object_detector"] = detection.ObjectDetector
    NodeRegistry._categories["object_detector"] = "detection"
    NodeRegistry._registry["embedder"] = embedding.Embedder
    NodeRegistry._categories["embedder"] = "embedding"
    NodeRegistry._registry["captioner"] = captioning.Captioner
    NodeRegistry._categories["captioner"] = "captioning"
    NodeRegistry._registry["pose_estimator"] = pose.PoseEstimator
    NodeRegistry._categories["pose_estimator"] = "pose"
    NodeRegistry._registry["remote"] = remote_node.RemoteNode
    NodeRegistry._categories["remote"] = "infrastructure"

    BackendRegistry._registry[("detection", "yolo")] = yolo.YoloBackend
    BackendRegistry._registry[("detection", "rtdetr")] = rtdetr.RTDetrBackend
    BackendRegistry._registry[("embedding", "siglip2")] = siglip2.SigLIP2Backend
    BackendRegistry._registry[("embedding", "clip")] = clip.CLIPBackend
    BackendRegistry._registry[("captioning", "smolvlm")] = smolvlm.SmolVLMBackend
    BackendRegistry._registry[("captioning", "llamacpp")] = llamacpp.LlamaCppBackend
    BackendRegistry._registry[("pose", "yolo_pose")] = yolo_pose.YoloPoseBackend


# Bootstrap on package import.
register_all()
