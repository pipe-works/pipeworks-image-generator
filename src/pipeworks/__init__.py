"""Pipeworks Image Generator - Multi-model AI image generation and editing."""

__version__ = "0.2.0"  # Breaking change: removed legacy ImageGenerator

from pipeworks.core.config import PipeworksConfig, config
from pipeworks.core.model_adapters import ModelAdapterBase, model_registry

# Import adapters to ensure they're registered
from pipeworks.core.adapters import QwenImageEditAdapter, ZImageTurboAdapter  # noqa: F401

__all__ = [
    "ModelAdapterBase",
    "model_registry",
    "PipeworksConfig",
    "config",
    "ZImageTurboAdapter",
    "QwenImageEditAdapter",
]
