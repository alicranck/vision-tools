from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.registry import NodeRegistry


def register_all() -> None:
    from vision_tools.nodes.io import remote_node
    from vision_tools.nodes.logic import dynamic_logic_node
    from vision_tools.nodes.model import captioning
    from vision_tools.nodes.model import detection
    from vision_tools.nodes.model import embedding
    from vision_tools.nodes.model import pose
    from vision_tools.nodes.state import buffer_node
    from vision_tools.nodes.state import track_node
    from vision_tools.nodes.utility import crop_node
    from vision_tools.nodes.utility import filter_node

    NodeRegistry._registry["object_detector"] = detection.ObjectDetector
    NodeRegistry._categories["object_detector"] = "detection"
    NodeRegistry._registry["embedder"] = embedding.Embedder
    NodeRegistry._categories["embedder"] = "embedding"
    NodeRegistry._registry["captioner"] = captioning.Captioner
    NodeRegistry._categories["captioner"] = "captioning"
    NodeRegistry._registry["pose_estimator"] = pose.PoseEstimator
    NodeRegistry._categories["pose_estimator"] = "pose"
    NodeRegistry._registry["remote"] = remote_node.RemoteNode
    NodeRegistry._categories["remote"] = "io"
    NodeRegistry._registry["dynamic_logic"] = dynamic_logic_node.DynamicLogicNode
    NodeRegistry._categories["dynamic_logic"] = "logic"
    NodeRegistry._registry["filter"] = filter_node.FilterNode
    NodeRegistry._categories["filter"] = "canonical"
    NodeRegistry._registry["track"] = track_node.TrackNode
    NodeRegistry._categories["track"] = "canonical"
    NodeRegistry._registry["crop"] = crop_node.CropNode
    NodeRegistry._categories["crop"] = "canonical"
    NodeRegistry._registry["buffer"] = buffer_node.BufferNode
    NodeRegistry._categories["buffer"] = "canonical"

    try:
        from vision_tools.backends.detection import rtdetr
        from vision_tools.backends.detection import yolo
        from vision_tools.backends.embedding import clip, siglip2
        from vision_tools.backends.captioning import llamacpp, smolvlm
        from vision_tools.backends.pose import yolo_pose
    except ImportError:
        return

    BackendRegistry._registry[("detection", "yolo")] = yolo.YoloBackend
    BackendRegistry._registry[("detection", "rtdetr")] = rtdetr.RTDetrBackend
    BackendRegistry._registry[("embedding", "siglip2")] = siglip2.SigLIP2Backend
    BackendRegistry._registry[("embedding", "clip")] = clip.CLIPBackend
    BackendRegistry._registry[("captioning", "smolvlm")] = smolvlm.SmolVLMBackend
    BackendRegistry._registry[("captioning", "llamacpp")] = llamacpp.LlamaCppBackend
    BackendRegistry._registry[("pose", "yolo_pose")] = yolo_pose.YoloPoseBackend


register_all()
