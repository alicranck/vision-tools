import numpy as np

from vision_tools.backends.captioning.smolvlm import SmolVLMBackend


class _DummyTensor:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _DummyProcessor:
    def __init__(self):
        self.last = None

    def apply_chat_template(self, *args, **kwargs):
        self.last = _DummyTensor()
        return self.last


def test_preprocess_uses_backend_device_cpu_by_default():
    backend = SmolVLMBackend()
    backend._processor = _DummyProcessor()
    backend._device = "cpu"

    out = backend.preprocess(np.zeros((8, 8, 3), dtype=np.uint8))
    assert isinstance(out, _DummyTensor)
    assert out.device == "cpu"


def test_preprocess_uses_backend_device_when_overridden():
    backend = SmolVLMBackend()
    backend._processor = _DummyProcessor()
    backend._device = "cuda"

    out = backend.preprocess(np.zeros((8, 8, 3), dtype=np.uint8))
    assert isinstance(out, _DummyTensor)
    assert out.device == "cuda"
