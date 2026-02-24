import os
from pathlib import Path

APP_DIR = Path(__file__).parent.parent
CONFIGS_DIR = APP_DIR / "core" / "configs"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vision_tools"
CACHE_DIR = Path(os.getenv("VISION_TOOLS_CACHE_DIR", str(DEFAULT_CACHE_DIR))).expanduser()
MODELS_CACHE_DIR = CACHE_DIR / "models"


def setup_cache_env() -> None:
    """Setup environment variables for common ML libraries to use the centralized cache."""
    # Ensure directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Hugging Face
    hf_cache = CACHE_DIR / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))

    # PyTorch/Torchvision
    torch_cache = CACHE_DIR / "torch"
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache))

    # Ultralytics (YOLO)
    yolo_dir = CACHE_DIR / "ultralytics"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_dir))

    # Try to set ultralytics settings directly if available
    try:
        from ultralytics import settings as yolo_settings
        yolo_settings.update({
            "weights_dir": str(MODELS_CACHE_DIR),
            "datasets_dir": str(CACHE_DIR / "datasets"),
            "runs_dir": str(CACHE_DIR / "runs"),
        })
    except Exception:
        pass