"""
Model registry — central catalog and factory for all prediction models.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from endforecast.models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    _registry: dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_cls: Type[BaseModel]) -> None:
        if name in cls._registry:
            logger.warning("Model '%s' already registered; overwriting.", name)
        cls._registry[name] = model_cls

    @classmethod
    def build(cls, name: str, params: Optional[dict[str, Any]] = None) -> BaseModel:
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name](params=params)

    @classmethod
    def list_registered(cls) -> list[str]:
        return sorted(cls._registry.keys())


def register_model(name: str):
    """Decorator to register a model class in the global registry."""
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        ModelRegistry.register(name, cls)
        return cls
    return decorator