from pathlib import Path


APP_DIR = Path(__file__).parent.parent
CONFIGS_DIR = APP_DIR / "core" / "configs"
CACHE_DIR = Path.home() / ".cache" / "vision_tools"