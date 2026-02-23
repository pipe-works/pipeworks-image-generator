"""Pipe-Works Image Generator — core generation engine.

This sub-package contains the two foundational modules of the image generator:

Modules
-------
config
    :class:`PipeworksConfig` — a Pydantic Settings class that reads all
    configuration from ``PIPEWORKS_*`` environment variables.  A global
    singleton ``config`` is instantiated at import time so that every other
    module can simply ``from pipeworks.core.config import config``.

model_manager
    :class:`ModelManager` — manages the lifecycle of a single HuggingFace
    Diffusers pipeline.  Handles lazy loading, model switching, CUDA memory
    cleanup, turbo-model guidance enforcement, and deterministic seeding.

The core package has **no dependency on HTTP or FastAPI** — it is a pure
generation engine that the API layer wraps with REST endpoints.

Exports
-------
For convenience the most-used symbols are re-exported here so callers can
write ``from pipeworks.core import ModelManager`` without reaching into
individual modules.
"""

from __future__ import annotations

from pipeworks.core.config import PipeworksConfig, config
from pipeworks.core.model_manager import ModelManager

__all__ = [
    "ModelManager",
    "PipeworksConfig",
    "config",
]
