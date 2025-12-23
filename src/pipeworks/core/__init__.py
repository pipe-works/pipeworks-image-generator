"""Core functionality for image generation.

This module provides the core components for the Pipeworks Image Generator:

- **Model Adapters**: Flexible multi-model support system
- **model_registry**: Registry for discovering and instantiating model adapters
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

3. **Support Utilities**:
   - prompt_builder.py: File-based prompt construction
   - tokenizer.py: Token analysis for prompts
   - gallery_browser.py: Image browsing and metadata display
   - favorites_db.py: SQLite-based favorites tracking
   - catalog_manager.py: Archive management for favorited images
   - character_conditions.py: Procedural character generation
   - facial_conditions.py: Facial signal generation (experimental)

Usage Example
-------------
Text-to-image generation:

    from pipeworks.core import model_registry, config

    # Instantiate Z-Image-Turbo adapter
    adapter = model_registry.instantiate("Z-Image-Turbo", config)

    # Generate and save image
    image, path = adapter.generate_and_save(
        prompt="a serene mountain landscape at sunset",
        width=1024,
        height=1024,
        num_inference_steps=9,
        seed=42
    )
    print(f"Saved to: {path}")

Image editing:

    from pipeworks.core import model_registry, config
    from PIL import Image

    # Instantiate Qwen-Image-Edit adapter
    editor = model_registry.instantiate("Qwen-Image-Edit", config)

    # Load base image
    base_image = Image.open("character.png")

    # Edit and save
    edited, path = editor.generate_and_save(
        input_image=base_image,
        instruction="change the sky to sunset colors",
        num_inference_steps=40,
        seed=42
    )
    print(f"Saved to: {path}")

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

__all__ = [
    "ModelAdapterBase",
    "model_registry",
    "PipeworksConfig",
    "config",
    "ZImageTurboAdapter",
    "QwenImageEditAdapter",
]
