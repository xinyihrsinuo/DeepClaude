"""Model Configuration Classes"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, root_validator, validator

from app.utils.logger import logger


class ProviderType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OPENAI_COMPATIBLE = "openai-compatible"


class Provider(BaseModel):
    name: str
    type: ProviderType
    base_url: str
    api_key: str

    # validate existed ProviderType
    @validator("type")
    def validate_provider_type(cls, v) -> ProviderType:
        if v not in ProviderType:
            logger.error(f"Invalid provider type: '{v}'")
            raise ValueError(
                f"Invalid provider type: '{v}'\nsupported provider types: {ProviderType}"
            )
        return v

    @root_validator(skip_on_failure=True)
    def validate_fields_not_none(cls, values):
        for field_name, value in values.items():
            if value is None:
                logger.error(f"Field '{field_name}' cannot be None")
                raise ValueError(f"Field '{field_name}' cannot be None")
        return values


class BaseModelConfig(BaseModel):
    name: str
    model_id: str
    provider: str
    context: int = Field(gt=0)
    max_tokens: int = Field(gt=0)

    @root_validator(skip_on_failure=True)
    def validate_fields_not_none(cls, values):
        for field_name, value in values.items():
            if value is None:
                logger.error(f"Field '{field_name}' cannot be None")
                raise ValueError(f"Field '{field_name}' cannot be None")
        return values


class DeepModelConfig(BaseModel):
    name: str
    reason_model: str
    answer_model: str
    is_origin_reasoning: bool = Field(default=True)

    @root_validator(skip_on_failure=True)
    def validate_fields_not_none(cls, values):
        for field_name, value in values.items():
            if value is None:
                logger.error(f"Field '{field_name}' cannot be None")
                raise ValueError(f"Field '{field_name}' cannot be None")
        return values


class ModelConfig(BaseModel):
    providers: List[Provider]
    base_models: List[BaseModelConfig]
    deep_models: List[DeepModelConfig]

    @validator("providers")
    def check_unique_provider_names(cls, providers):
        names = [p.name for p in providers]
        if len(names) != len(set(names)):
            logger.error(f"Provider '{names}' duplicated")
            raise ValueError("Provider names must be unique")
        return providers

    @validator("base_models")
    def check_unique_base_model_names(cls, base_models):
        names = [p.name for p in base_models]
        if len(names) != len(set(names)):
            logger.error(f"Base model '{names}' duplicated")
            raise ValueError("Base model names must be unique")
        return base_models

    @root_validator(skip_on_failure=True)
    def validate_fields_not_none(cls, values):
        for field_name, value in values.items():
            if value is None:
                logger.error(f"Field '{field_name}' cannot be None")
                raise ValueError(f"Field '{field_name}' cannot be None")
        return values

    @root_validator(skip_on_failure=True)
    def check_reference_existence(cls, values):
        providers: List[Provider] = values.get("providers", [])
        base_models: List[BaseModelConfig] = values.get("base_models", [])
        provider_names = {p.name for p in providers}
        for model in base_models:
            if model.provider not in provider_names:
                logger.error(
                    f"Provider '{model.provider}' of BaseModel '{model.name}' not found"
                )
                raise ValueError(f"Provider '{model.provider}' not found")

        deep_models: List[DeepModelConfig] = values.get("deep_models", [])
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
        return values


model_config = None


def get_model_config() -> ModelConfig:
    if model_config is None:
        raise ValueError("Model config not initialized")
    return model_config


def set_model_config(data):
    global model_config
    model_config = ModelConfig(**data)
