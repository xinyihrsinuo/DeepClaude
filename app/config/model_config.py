"""Model Configuration Classes"""

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field, model_validator, validator

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

    # validate existed ProviderType
    @validator("type")
    def _validate_provider_type(cls, v) -> ProviderType:
        if v not in ProviderType:
            logger.error(f"Invalid provider type: '{v}'")
            raise ValueError(
                f"Invalid provider type: '{v}'\nsupported provider types: {ProviderType}"
            )
        return v

    @model_validator(mode="after")
    def _validate_fields_not_none(self):
        for field_name, value in self.__dict__.items():
            if value is None:
                logger.error(f"Field '{field_name}' cannot be None")
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
                logger.error(f"Field '{field_name}' cannot be None")
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
                logger.error(f"Field '{field_name}' cannot be None")
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

    @validator("providers")
    def _check_unique_provider_names(cls, providers):
        names = [p.name for p in providers]
        if len(names) != len(set(names)):
            logger.error(f"Provider '{names}' duplicated")
            raise ValueError("Provider names must be unique")
        return providers

    @validator("base_models")
    def _check_unique_base_model_names(cls, base_models):
        names = [p.name for p in base_models]
        if len(names) != len(set(names)):
            logger.error(f"Base model '{names}' duplicated")
            raise ValueError("Base model names must be unique")
        return base_models

    @model_validator(mode="after")
    def _validate_fields_not_none(self):
        for field_name, value in self.__dict__.items():
            if value is None:
                logger.error(f"Field '{field_name}' cannot be None")
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
                    f"Provider '{model.provider}' of BaseModel '{model.name}' not found"
                )
                raise ValueError(f"Provider '{model.provider}' not found")

        deep_models: List[DeepModelConfig] = self.deep_models
        base_model_names = [m.name for m in base_models]
        for model in deep_models:
            if model.reason_model not in base_model_names:
                logger.error(
                    f"Reason model '{model.reason_model}' of DeepModel '{model.name}' not found"
                )
                raise ValueError(f"Reason model '{model.reason_model}' not found")
            if model.answer_model not in base_model_names:
                logger.error(
                    f"Base model '{model.answer_model}' of DeepModel '{model.name}' not found"
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
            logger.error(f"Deep model '{name}' not found")
            raise ValueError(f"Deep model '{name}' not found")
        return model

    def get_provider(self, name: str) -> ProviderConfig:
        provider = self._provider_map.get(name)
        if not provider:
            logger.error(f"Provider '{name}' not found")
            raise ValueError(f"Provider '{name}' not found")
        return provider

    def get_base_model(self, name: str) -> BaseModelConfig:
        model = self._base_model_map.get(name)
        if not model:
            logger.error(f"Model '{name}' not found")
            raise ValueError(f"Model '{name}' not found")
        return model

    def get_model_request_info(
        self, model_name: str
    ) -> tuple[str, str, str, ProviderType]:
        base_model = self.get_base_model(model_name)
        provider = self.get_provider(base_model.provider)

        return (
            base_model.model_id,
            provider.base_url,
            provider.api_key,
            provider.type,
        )


model_config = None


def get_model_config() -> ModelConfig:
    if model_config is None:
        raise ValueError("Model config not initialized")
    return model_config


def set_model_config(data):
    global model_config
    model_config = ModelConfig(**data)
