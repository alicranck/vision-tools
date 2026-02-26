import os
from pathlib import Path

APP_DIR = Path(__file__).parent.parent
CONFIGS_DIR = APP_DIR / "core" / "configs"

CACHE_ENV = "VISION_TOOLS_CACHE_DIR"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vision_tools"


def get_cache_dir() -> Path:
    """Resolve the root cache directory.
    """
    configured_root = os.getenv(CACHE_ENV)
    if configured_root:
        return Path(configured_root).expanduser()

    return DEFAULT_CACHE_DIR


def get_models_cache_dir(cache_dir: Path | None = None) -> Path:
    """Resolve the model-checkpoint cache directory."""
    return (cache_dir or get_cache_dir()) / "models"


CACHE_DIR = get_cache_dir()
MODELS_CACHE_DIR = get_models_cache_dir(CACHE_DIR)


def setup_cache_env() -> None:
    """Setup environment variables for common ML libraries to use the centralized cache."""
    explicit_cache_env = CACHE_ENV in os.environ
    cache_dir = get_cache_dir()
    models_cache_dir = get_models_cache_dir(cache_dir)

    # Publish resolved paths for downstream code.
    os.environ.setdefault(CACHE_ENV, str(cache_dir))

    # Ensure directories exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    models_cache_dir.mkdir(parents=True, exist_ok=True)

    # Hugging Face
    hf_cache = cache_dir / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))

    # PyTorch/Torchvision
    torch_cache = cache_dir / "torch"
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache))

    # Ultralytics (YOLO)
    yolo_dir = cache_dir / "ultralytics"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    if explicit_cache_env or not os.getenv("YOLO_CONFIG_DIR"):
        os.environ["YOLO_CONFIG_DIR"] = str(yolo_dir)

    # Try to set ultralytics settings directly if available
    try:
        from ultralytics import settings as yolo_settings
        yolo_settings.update({
            "weights_dir": str(models_cache_dir),
            "datasets_dir": str(cache_dir / "datasets"),
            "runs_dir": str(cache_dir / "runs"),
        })
    except Exception:
        pass
