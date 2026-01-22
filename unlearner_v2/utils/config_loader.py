import yaml
from pathlib import Path


def load_config(config_dir: str) -> dict:
    """加载并合并所有配置文件"""
    config_dir = Path(config_dir)
    config = {}
    for name in ["config", "model", "data", "train"]:
        path = config_dir / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)
    return config
