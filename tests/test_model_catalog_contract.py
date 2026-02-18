from vision_tools.capabilities import list_capabilities
from vision_tools.core.config import DeviceTarget, InferenceTask, ModelIntent, ModelSize
from vision_tools.runtime.model_catalog import ModelCatalog


def test_model_catalog_resolves_detector_intent() -> None:
    spec = ModelCatalog.resolve(
        ModelIntent(
            task=InferenceTask.DETECTION,
            model_family="yolo",
            size=ModelSize.SMALL,
            device=DeviceTarget.CPU,
        )
    )

    assert spec.backend_task == "detection"
    assert spec.backend_model == "yolo"
    assert spec.checkpoint_id
    assert spec.runtime in {"openvino", "cpu", "auto", "pytorch"}


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
