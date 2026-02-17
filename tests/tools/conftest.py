"""
Shared fixtures for vision-tools integration tests.

All tool fixtures are session-scoped to avoid reloading heavy ML models
between tests. The test_image fixture loads the shared test asset once.
"""
import os
import pytest
import yaml
import numpy as np

from vision_tools.utils.image_utils import load_image_opencv
from vision_tools.utils.locations import CONFIGS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_name: str) -> dict:
    """Load a tool config YAML by name."""
    cfg_path = CONFIGS_DIR / f"{config_name}.yaml"
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

# Resolve the cached YOLO detection model (auto-scaling may pick an
# uncached variant when running on CPU)
_yolo_cache = os.path.join(os.path.expanduser("~"), ".cache", "vision_tools", "yoloe-11s-seg.pt")
CACHED_YOLO_MODEL = _yolo_cache if os.path.exists(_yolo_cache) else None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_image_path():
    """Path to the shared test image (person + car scene)."""
    return os.path.join(ASSETS_DIR, "test_image.png")


@pytest.fixture(scope="session")
def test_image(test_image_path):
    """Load the test image as a numpy array (RGB, uint8)."""
    return load_image_opencv(test_image_path)


@pytest.fixture(scope="session")
def detector():
    """Loaded OpenVocabularyDetector with person + car vocabulary."""
    from vision_tools.core.tools.detection import OpenVocabularyDetector
    config = load_config("ov_detection")
    config["vocabulary"] = ["person", "car"]
    model_id = CACHED_YOLO_MODEL or config.get("model")
    det = OpenVocabularyDetector(model_id, config)
    return det


@pytest.fixture(scope="session")
def embedder():
    """Loaded SigLIP2Embedder."""
    from vision_tools.core.tools.embedder import SigLIP2Embedder
    config = load_config("embedding")
    emb = SigLIP2Embedder(config.get("model"), config)
    return emb


@pytest.fixture(scope="session")
def pose_estimator():
    """Loaded PoseEstimator."""
    from vision_tools.core.tools.pose_estimation import PoseEstimator
    config = load_config("pose_estimation")
    est = PoseEstimator(config.get("model"), config)
    return est
