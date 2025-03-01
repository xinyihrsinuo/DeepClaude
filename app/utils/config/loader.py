"""Config loaders for DeepClaude."""

from pathlib import Path
from typing import Any, Dict, List

import yaml

from app.utils.config.manager import ModelConfigManager
from app.utils.logger import logger


SHOWN_MODEL_CACHE = None


def load_model_config(yaml_path: Path):
    """从配置文件加载模型配置"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        logger.error("Model config file: %s is empty", yaml_path)
        raise ValueError("Config file is empty")
    ModelConfigManager.set_model_config(data)


def load_shown_model_config() -> Dict[str, List[Dict[str, Any]]]:
    """加载模型配置

    Returns:
        Dict[str, List[Dict[str, Any]]]: 模型配置字典
    """
    global SHOWN_MODEL_CACHE

    if SHOWN_MODEL_CACHE is not None:
        return SHOWN_MODEL_CACHE
    config_path = Path(__file__).parent.parent.parent / "shown_models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        SHOWN_MODEL_CACHE = yaml.safe_load(f)
        return SHOWN_MODEL_CACHE
