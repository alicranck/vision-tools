import sys
from pathlib import Path
from types import SimpleNamespace

from vision_tools.backends.detection.yolo import OpenVocabularyYoloBackend
from vision_tools.backends.pose.yolo_pose import YoloPoseBackend


class _FakeDetections:
    def __init__(self):
        self.tracker_id = []
        self.xyxy = []
        self.class_id = []
        self.confidence = []


class _FakeTracker:
    def update(self, detections):
        return detections


class _FakeModel:
    def __init__(self):
        self.last_kwargs = {}

    def predict(self, inputs, **kwargs):
        self.last_kwargs = dict(kwargs)
        return [SimpleNamespace(names={})]


def test_infer_uses_configured_imgsz_and_conf(monkeypatch):
    backend = OpenVocabularyYoloBackend()
    backend.configure({"imgsz": 1024, "conf_threshold": 0.73})
    backend._tracker = _FakeTracker()

    fake_sv = SimpleNamespace(
        Detections=SimpleNamespace(from_ultralytics=lambda _: _FakeDetections())
    )
    monkeypatch.setitem(sys.modules, "supervision", fake_sv)
    monkeypatch.setitem(
        sys.modules,
        "vision_tools.utils.tracking",
        SimpleNamespace(BoxKalmanFilter=object),
    )

    model = _FakeModel()
    backend.infer(model, inputs="frame")

    assert model.last_kwargs["imgsz"] == 1024
    assert model.last_kwargs["conf"] == 0.73


def test_resolve_checkpoint_path_maps_bare_pt_to_weights_dir():
    resolved = OpenVocabularyYoloBackend._resolve_checkpoint_path(
        "yoloe-11s-seg.pt",
        {"weights_dir": Path("/tmp/cache/models")},
    )
    assert resolved == "/tmp/cache/models/yoloe-11s-seg.pt"


def test_resolve_checkpoint_path_keeps_absolute_path():
    resolved = OpenVocabularyYoloBackend._resolve_checkpoint_path(
        "/models/custom.pt",
        {"weights_dir": Path("/tmp/cache/models")},
    )
    assert resolved == "/models/custom.pt"


def test_pose_resolve_checkpoint_path_maps_bare_pt_to_weights_dir():
    resolved = YoloPoseBackend._resolve_checkpoint_path(
        "yolo11n-pose.pt",
        {"weights_dir": Path("/tmp/cache/models")},
    )
    assert resolved == "/tmp/cache/models/yolo11n-pose.pt"
