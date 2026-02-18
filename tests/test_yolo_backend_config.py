import sys
from types import SimpleNamespace

from vision_tools.backends.detection.yolo import YoloBackend


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
    backend = YoloBackend()
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
