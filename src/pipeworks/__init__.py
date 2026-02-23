"""Pipe-Works Image Generator — AI image generation with FastAPI backend.

This package provides a complete image generation system built on HuggingFace
Diffusers pipelines, served through a FastAPI REST API with a vanilla JS
frontend.  The architecture follows a simple layered design:

Layers
------
core
    Configuration management (:class:`PipeworksConfig`) and diffusion pipeline
    lifecycle management (:class:`ModelManager`).  The core layer has no
    knowledge of HTTP or the API — it deals only with configuration, model
    loading, and image generation.

api
    FastAPI application with REST endpoints for generation, gallery management,
    prompt compilation, and static asset serving.  The API layer depends on
    ``core`` for model inference and configuration.

static / templates
    Vanilla HTML/CSS/JS frontend assets served by FastAPI's ``StaticFiles``
    mount and ``HTMLResponse``.  No build step or template engine required.

Quick Start
-----------
::

    from pipeworks.core.config import config
    from pipeworks.core.model_manager import ModelManager

    mgr = ModelManager(config)
    mgr.load_model("Tongyi-MAI/Z-Image-Turbo")

    image = mgr.generate(
        prompt="a goblin workshop",
        width=1024, height=1024,
        steps=4, guidance_scale=0.0, seed=42,
    )

Or launch the full web application::

    pipeworks          # CLI entry point
    # → FastAPI server on http://0.0.0.0:7860

Exports
-------
The package re-exports the most commonly used symbols for convenience:

- :class:`ModelManager` — pipeline lifecycle management.
- :class:`PipeworksConfig` — Pydantic Settings configuration class.
- ``config`` — the global configuration singleton.
- ``__version__`` — current package version string.
"""

from __future__ import annotations

from importlib.metadata import version

# ---------------------------------------------------------------------------
# Package version — read from pyproject.toml via importlib.metadata.
# This is the single source of truth managed by release-please, which bumps
# the version field in pyproject.toml on each release.  No manual edits
# needed here.
# ---------------------------------------------------------------------------
__version__: str = version("pipeworks-image-generator")

# ---------------------------------------------------------------------------
# Convenience re-exports.
# These allow callers to write ``from pipeworks import ModelManager`` instead
# of the fully-qualified ``from pipeworks.core.model_manager import ...``.
# ---------------------------------------------------------------------------
from pipeworks.core.config import PipeworksConfig, config
from pipeworks.core.model_manager import ModelManager

__all__ = [
    "ModelManager",
    "PipeworksConfig",
    "config",
    "__version__",
]
