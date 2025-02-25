"""Config loaders for DeepClaude."""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from app.config.model_config import set_model_config
from app.utils.logger import logger


def load_model_config(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if not data:
        logger.error(f"Model config file: {yaml_path} is empty")
        raise ValueError("Config file is empty")
    set_model_config(data)


shown_model_cache = None


def load_shown_model_config() -> Dict[str, List[Dict[str, Any]]]:
    """加载模型配置

    Returns:
        Dict[str, List[Dict[str, Any]]]: 模型配置字典
    """
    global shown_model_cache

    if shown_model_cache is not None:
        return shown_model_cache
    config_path = Path(__file__).parent.parent.parent / "shown_models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        shown_model_cache = yaml.safe_load(f)
        return shown_model_cache
