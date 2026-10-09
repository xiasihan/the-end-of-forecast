"""
Model abstraction — uniform interface and registry for prediction models.
"""

from endforecast.models.base import BaseModel
from endforecast.models.registry import ModelRegistry, register_model

__all__ = ["BaseModel", "ModelRegistry", "register_model"]