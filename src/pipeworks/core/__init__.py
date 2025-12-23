"""Core functionality for image generation.

This module provides the core components for the Pipeworks Image Generator:

- **Model Adapters**: Flexible multi-model support system
- **model_registry**: Registry for discovering and instantiating model adapters
- **ImageGenerator**: Legacy pipeline wrapper (use model adapters instead)
- **PipeworksConfig**: Configuration management using Pydantic Settings
- **config**: Global configuration instance (loads from environment variables)

The core module is designed to be the primary interface for image generation,
supporting multiple AI models through a unified adapter pattern.

Architecture Overview
---------------------
The core module follows a layered architecture:

1. **Configuration Layer** (config.py):
   - Environment-based configuration using Pydantic Settings
   - All settings prefixed with PIPEWORKS_ in .env files
   - Automatic directory creation and path resolution

2. **Model Adapter Layer** (model_adapters.py, adapters/):
   - Unified interface for different AI models
   - Model-specific implementations (Z-Image-Turbo, Qwen-Image-Edit, etc.)
   - Plugin lifecycle integration
   - Registry pattern for model discovery

3. **Legacy Pipeline Layer** (pipeline.py):
   - Original Z-Image-Turbo wrapper (deprecated in favor of adapters)
   - Maintained for backward compatibility

4. **Support Utilities**:
   - prompt_builder.py: File-based prompt construction
   - tokenizer.py: Token analysis for prompts
   - gallery_browser.py: Image browsing and metadata display
   - favorites_db.py: SQLite-based favorites tracking
   - catalog_manager.py: Archive management for favorited images

Usage Example
-------------
Using model adapters (recommended):

    from pipeworks.core import model_registry, config

    # Text-to-image generation
    adapter = model_registry.instantiate("Z-Image-Turbo", config)
    image, path = adapter.generate_and_save(
        prompt="a beautiful landscape",
        seed=42
    )

    # Image editing
    editor = model_registry.instantiate("Qwen-Image-Edit", config)
    edited, path = editor.generate_and_save(
        input_image=base_image,
        instruction="change sky to sunset",
        seed=42
    )

Legacy usage (backward compatible):

    from pipeworks.core import ImageGenerator

    generator = ImageGenerator()
    image, path = generator.generate_and_save(
        prompt="a beautiful landscape",
        seed=42
    )

See Also
--------
- ModelAdapterBase: Base class for model adapters
- model_registry: Registry for model discovery
- PipeworksConfig: Configuration options and environment variables
"""

# Import adapters to ensure they're registered
# This must happen after model_registry is imported
from pipeworks.core.adapters import QwenImageEditAdapter, ZImageTurboAdapter  # noqa: F401
from pipeworks.core.config import PipeworksConfig, config
from pipeworks.core.model_adapters import ModelAdapterBase, model_registry
from pipeworks.core.pipeline import ImageGenerator

__all__ = [
    "ImageGenerator",  # Legacy
    "ModelAdapterBase",
    "model_registry",
    "PipeworksConfig",
    "config",
]
