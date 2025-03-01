"""Model Configuration Classes"""

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field, model_validator

from app.utils.logger import logger


class ProviderType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OPENAI_COMPATIBLE = "openai-compatible"


class ProviderConfig(BaseModel):
    name: str
    type: ProviderType
    base_url: str
    api_key: str
    use_proxy: bool

    # validate existed ProviderType
    @model_validator(mode="after")
    def _validate_provider_type(self) -> "ProviderConfig":
        if self.type not in ProviderType:
            logger.error("Invalid provider type: '%s'", self.type)
            raise ValueError(
                f"Invalid provider type: '{self.type}'\nsupported provider types: {ProviderType}"
            )
        return self

    @model_validator(mode="after")
    def _validate_base_url(self) -> "ProviderConfig":
        if not self.base_url.startswith("http"):
            logger.warning(
                "Base URL '%s' does not start with http or https", self.base_url
            )
        return self

    @model_validator(mode="after")
    def _validate_fields_not_none(self):
        for field_name, value in self.__dict__.items():
            if value is None:
                logger.error("Field '%s' cannot be None", field_name)
                raise ValueError(f"Field '{field_name}' cannot be None")
        return self


class BaseModelConfig(BaseModel):
    name: str
    model_id: str
    provider: str
    context: int = Field(gt=0)
    max_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_fields_not_none(self):
        for field_name, value in self.__dict__.items():
            if value is None:
                logger.error("Field '%s' cannot be None", field_name)
                raise ValueError(f"Field '{field_name}' cannot be None")
        return self


class DeepModelConfig(BaseModel):
    name: str
    reason_model: str
    answer_model: str
    is_origin_reasoning: bool = Field(default=True)

    @model_validator(mode="after")
    def _validate_fields_not_none(self):
        for field_name, value in self.__dict__.items():
            if value is None:
                logger.error("Field '%s' cannot be None", field_name)
                raise ValueError(f"Field '{field_name}' cannot be None")
        return self


class ModelConfig(BaseModel):
    providers: List[ProviderConfig]
    base_models: List[BaseModelConfig]
    deep_models: List[DeepModelConfig]
    _provider_map: Dict[str, ProviderConfig]
    _base_model_map: Dict[str, BaseModelConfig]
    _deep_model_map: Dict[str, DeepModelConfig]
    _context_map: Dict[str, int]

    @model_validator(mode="after")
    def _check_unique_provider_names(self) -> "ModelConfig":
        names = [p.name for p in self.providers]
        if len(names) != len(set(names)):
            logger.error("Provider '%s' duplicated", names)
            raise ValueError("Provider names must be unique")
        return self

    @model_validator(mode="after")
    def _check_unique_base_model_names(self) -> "ModelConfig":
        names = [p.name for p in self.base_models]
        if len(names) != len(set(names)):
            logger.error("Base model '%s' duplicated", names)
            raise ValueError("Base model names must be unique")
        return self

    @model_validator(mode="after")
    def _validate_fields_not_none(self):
        for field_name, value in self.__dict__.items():
            if value is None:
                logger.error("Field '%s' cannot be None", field_name)
                raise ValueError(f"Field '{field_name}' cannot be None")
        return self

    @model_validator(mode="after")
    def _check_reference_existence(self):
        providers: List[ProviderConfig] = self.providers
        base_models: List[BaseModelConfig] = self.base_models
        provider_names = {p.name for p in providers}
        for model in base_models:
            if model.provider not in provider_names:
                logger.error(
                    "Provider '%s' of BaseModel '%s' not found",
                    model.provider,
                    model.name,
                )
                raise ValueError(f"Provider '{model.provider}' not found")

        deep_models: List[DeepModelConfig] = self.deep_models
        base_model_names = [m.name for m in base_models]
        for model in deep_models:
            if model.reason_model not in base_model_names:
                logger.error(
                    "Reason model '%s' of DeepModel '%s' not found",
                    model.reason_model,
                    model.name,
                )
                raise ValueError(f"Reason model '{model.reason_model}' not found")
            if model.answer_model not in base_model_names:
                logger.error(
                    "Base model '%s' of DeepModel '%s' not found",
                    model.answer_model,
                    model.name,
                )
                raise ValueError(f"Base model '{model.answer_model}' not found")
        return self

    @model_validator(mode="after")
    def _build_maps(self):
        """Build maps as index for providers, base models, and contexts. Optimized for performance"""
        providers: List[ProviderConfig] = self.providers
        base_models: List[BaseModelConfig] = self.base_models
        self._provider_map = {p.name: p for p in providers}
        self._base_model_map = {m.name: m for m in base_models}
        self._deep_model_map = {m.name: m for m in self.deep_models}
        self._context_map = {}

        for model in self.deep_models:
            reason_context = self.get_base_model(model.reason_model).context
            answer_context = self.get_base_model(model.answer_model).context
            # Select the maximum context between reason and answer models as the context for the deep model
            self._context_map[model.name] = max(reason_context, answer_context)

        return self

    def get_deep_model(self, name: str) -> DeepModelConfig:
        model = self._deep_model_map.get(name)
        if not model:
            raise ValueError(f"Deep model '{name}' not found")
        return model

    def get_provider(self, name: str) -> ProviderConfig:
        provider = self._provider_map.get(name)
        if not provider:
            logger.error("Provider '%s' not found", name)
            raise ValueError(f"Provider '{name}' not found")
        return provider

    def get_base_model(self, name: str) -> BaseModelConfig:
        model = self._base_model_map.get(name)
        if not model:
            logger.error("Model '%s' not found", name)
            raise ValueError(f"Model '{name}' not found")
        return model

    def get_model_request_info(
        self, model_name: str
    ) -> tuple[str, str, str, ProviderType, bool]:
        base_model = self.get_base_model(model_name)
        provider = self.get_provider(base_model.provider)

        return (
            base_model.model_id,
            provider.base_url,
            provider.api_key,
            provider.type,
            provider.use_proxy,
        )


MODEL_CONFIG = None


def get_model_config() -> ModelConfig:
    """Get MODEL_CONFIG"""
    if MODEL_CONFIG is None:
        raise ValueError("Model config not initialized")
    return MODEL_CONFIG


def set_model_config(data):
    """Set MODEL_CONFIG"""
    global MODEL_CONFIG
    MODEL_CONFIG = ModelConfig(**data)
