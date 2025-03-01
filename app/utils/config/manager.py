"""配置管理器, 管理分发全局配置"""
from typing import Optional
from app.config.model_config import ModelConfig


class ModelConfigManager:
    """模型配置管理"""
    _instance: Optional["ModelConfigManager"] = None
    _model_config: Optional[ModelConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelConfigManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def set_model_config(cls, data):
        """Set MODEL_CONFIG"""
        cls._model_config = ModelConfig(**data)

    @classmethod
    def get_model_config(cls) -> ModelConfig:
        """Get MODEL_CONFIG"""
        if cls._model_config is None:
            raise ValueError("Model config not initialized")
        return cls._model_config
