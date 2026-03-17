from vision_tools.capabilities import list_capabilities
from vision_tools.core.config import DeviceTarget, InferenceTask, ModelIntent, ModelSize
from vision_tools.runtime.model_catalog import ModelCatalog


def test_model_catalog_resolves_detector_intent() -> None:
    spec = ModelCatalog.resolve(
        ModelIntent(
            task=InferenceTask.DETECTION,
            model_family="yolo_detector",
            size=ModelSize.SMALL,
            device=DeviceTarget.CPU,
        )
    )

    assert spec.backend_task == "detection"
    assert spec.backend_model == "yolo_detector"
    assert spec.checkpoint_id
    assert spec.runtime in {"openvino", "cpu", "auto", "pytorch"}


def test_model_catalog_resolves_open_vocab_detector_intent() -> None:
    spec = ModelCatalog.resolve(
        ModelIntent(
            task=InferenceTask.OPEN_VOCAB_DETECTION,
            model_family="yolo",
            size=ModelSize.SMALL,
            device=DeviceTarget.CPU,
        )
    )

    assert spec.backend_task == "open_vocab_detection"
    assert spec.backend_model == "yolo"
    assert spec.checkpoint_id


def test_model_catalog_resolves_embedder_intent() -> None:
    spec = ModelCatalog.resolve(
        ModelIntent(
            task=InferenceTask.EMBEDDING,
            model_family="siglip2",
            size=ModelSize.MEDIUM,
            device=DeviceTarget.CPU,
        )
    )

    assert spec.backend_task == "embedding"
    assert spec.backend_model == "siglip2"
    assert spec.checkpoint_id == "google/siglip2-base-patch16-384"


def test_model_catalog_resolves_captioning_intent() -> None:
    spec = ModelCatalog.resolve(
        ModelIntent(
            task=InferenceTask.CAPTIONING,
            model_family="smolvlm",
            size=ModelSize.SMALL,
            device=DeviceTarget.CPU,
        )
    )

    assert spec.backend_task == "captioning"
    assert spec.backend_model == "smolvlm"
    assert spec.checkpoint_id == "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF"


def test_model_catalog_resolves_pose_intent() -> None:
    spec = ModelCatalog.resolve(
        ModelIntent(
            task=InferenceTask.POSE,
            model_family="yolo_pose",
            size=ModelSize.SMALL,
            device=DeviceTarget.CPU,
        )
    )

    assert spec.backend_task == "pose"
    assert spec.backend_model == "yolo_pose"
    assert spec.checkpoint_id == "yolo11n-pose.pt"


def test_model_catalog_rejects_unknown_family() -> None:
    try:
        ModelCatalog.resolve(
            ModelIntent(
                task=InferenceTask.DETECTION,
                model_family="does_not_exist",
                size=ModelSize.SMALL,
                device=DeviceTarget.CPU,
            )
        )
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_capabilities_exposes_model_catalog() -> None:
    caps = list_capabilities()
    assert "nodes" in caps
    assert "backends" in caps
    assert "model_catalog" in caps
    assert "detection" in caps["model_catalog"]
    assert "embedding" in caps["model_catalog"]
    assert "captioning" in caps["model_catalog"]
    assert "pose" in caps["model_catalog"]
