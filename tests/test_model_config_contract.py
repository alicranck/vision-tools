from vision_tools.nodes.model.captioning import Captioner, CaptionerConfig
from vision_tools.nodes.model.classification import Classifier, ClassifierConfig
from vision_tools.nodes.model.detection import DetectionNodeConfig, Detector
from vision_tools.nodes.model.embedding import Embedder, EmbedderConfig
from vision_tools.nodes.model.pose import PoseEstimator, PoseEstimatorConfig
from vision_tools.nodes.model.segmentation import Segmenter, SegmenterConfig


def test_catalog_backed_model_config_schemas_use_model_family_contract() -> None:
    cases = [
        (DetectionNodeConfig, {"task", "model_family", "size", "device"}),
        (ClassifierConfig, {"task", "model_family", "size", "device"}),
        (SegmenterConfig, {"task", "model_family", "size", "device"}),
        (EmbedderConfig, {"task", "model_family", "size", "device"}),
        (CaptionerConfig, {"task", "model_family", "size", "device"}),
        (PoseEstimatorConfig, {"task", "model_family", "size", "device"}),
    ]

    for schema, required_fields in cases:
        field_names = set(schema.model_fields.keys())
        assert required_fields.issubset(field_names)
        assert "model" not in field_names


def test_catalog_backed_nodes_resolve_checkpoint_id_from_model_family() -> None:
    cases = [
        (
            Detector,
            {"model_family": "yolo_detector", "size": "small", "device": "cpu"},
            "yolo11s.pt",
        ),
        (
            Classifier,
            {"model_family": "yolo_cls", "size": "small", "device": "cpu"},
            "yolo11n-cls.pt",
        ),
        (
            Segmenter,
            {"model_family": "yolo_seg", "size": "small", "device": "cpu"},
            "yolo11n-seg.pt",
        ),
        (
            Embedder,
            {"model_family": "siglip2", "size": "medium", "device": "cpu"},
            "google/siglip2-base-patch16-384",
        ),
        (
            Captioner,
            {"model_family": "smolvlm", "size": "small", "device": "cpu"},
            "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
        ),
        (
            PoseEstimator,
            {"model_family": "yolo_pose", "size": "small", "device": "cpu"},
            "yolo11n-pose.pt",
        ),
    ]

    for node_cls, config, expected_checkpoint in cases:
        node = node_cls(config=config)
        assert node.config["checkpoint_id"] == expected_checkpoint
        assert "model" not in node.config
