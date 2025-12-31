import yaml
from vision_tools.utils.locations import CONFIGS_DIR


def load_config(config_name: str) -> dict:
    cfg_path = CONFIGS_DIR / f"{config_name}.yaml"
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return config