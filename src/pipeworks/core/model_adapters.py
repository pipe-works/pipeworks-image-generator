"""Base classes and registry for model adapters.

This module provides the foundation for supporting multiple AI models in Pipeworks.
Each model (Z-Image-Turbo, Qwen-Image-Edit, etc.) has its own adapter that implements
a common interface while handling model-specific requirements.

Model Adapter Pattern
---------------------
The adapter pattern allows workflows to work with different models through a unified
interface. Each adapter encapsulates:
- Model loading and unloading
- Model-specific parameter handling
- Pipeline-specific optimizations
- Plugin lifecycle integration

Model Types
-----------
Different model types support different generation modes:
- **text-to-image**: Generate images from text prompts (e.g., Z-Image-Turbo)
- **image-edit**: Edit existing images with instructions (e.g., Qwen-Image-Edit)
- **img2img**: Transform images with prompts (e.g., SDXL img2img)
- **inpainting**: Fill masked regions of images

Usage Example
-------------
Using the registry to instantiate and use a model adapter:

    >>> from pipeworks.core.model_adapters import model_registry
    >>> from pipeworks.core.config import config
    >>>
    >>> # Get available models
    >>> print(model_registry.list_available())
    ['Z-Image-Turbo', 'Qwen-Image-Edit']
    >>>
    >>> # Instantiate a text-to-image model
    >>> adapter = model_registry.instantiate("Z-Image-Turbo", config)
    >>> image = adapter.generate(
    ...     prompt="a fantasy landscape",
    ...     seed=42
    ... )
    >>>
    >>> # Instantiate an image editing model
    >>> editor = model_registry.instantiate("Qwen-Image-Edit", config)
    >>> edited = editor.generate(
    ...     input_image=base_image,
    ...     instruction="change sky to sunset",
    ...     seed=42
    ... )

Plugin Integration
------------------
Model adapters support the plugin system through lifecycle hooks:
- Plugins are passed to adapters during instantiation
- Adapters call plugin hooks at appropriate lifecycle points
- Plugin hooks work consistently across all model types

See Also
--------
- PluginBase: Plugin system documentation
- WorkflowBase: Workflow system documentation
- PipeworksConfig: Configuration options
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from pipeworks.plugins.base import PluginBase

from .config import PipeworksConfig

logger = logging.getLogger(__name__)


class ModelAdapterBase(ABC):
    """Abstract base class for all model adapters.

    Model adapters provide a unified interface for different AI models, handling
    model-specific requirements and parameters while presenting a consistent API
    to workflows and application code.

    Each adapter must implement:
    - Model loading and unloading
    - Image generation (with model-specific parameters)
    - Model state management
    - Plugin lifecycle integration

    Attributes
    ----------
    name : str
        Human-readable name of the model (e.g., "Z-Image-Turbo")
    description : str
        Brief description of the model's capabilities
    model_type : str
        Type of generation this model supports
    config : PipeworksConfig
        Configuration object containing model settings
    plugins : list[PluginBase]
        List of active plugin instances

    Notes
    -----
    - Adapters should implement lazy loading (load model on first use)
    - All adapters must be thread-safe for multi-workflow scenarios
    - Plugin hooks should be called at consistent lifecycle points
    - Adapters should handle model-specific constraints (e.g., guidance_scale=0 for Turbo)

    Examples
    --------
    Creating a custom adapter:

        >>> class MyModelAdapter(ModelAdapterBase):
        ...     name = "My Model"
        ...     description = "Custom model for specific tasks"
        ...     model_type = "text-to-image"
        ...
        ...     def load_model(self):
        ...         # Load your model here
        ...         pass
        ...
        ...     def generate(self, prompt: str, **kwargs) -> Image.Image:
        ...         # Generate image here
        ...         pass
        ...
        ...     # Implement other required methods...
        >>>
        >>> # Register the adapter
        >>> from pipeworks.core.model_adapters import model_registry
        >>> model_registry.register(MyModelAdapter)
    """

    name: str = "Base Model Adapter"
    description: str = "Base class for model adapters"
    model_type: Literal["text-to-image", "image-edit", "img2img", "inpainting"] = "text-to-image"
    version: str = "0.1.0"

    def __init__(self, config: PipeworksConfig, plugins: list[PluginBase] | None = None) -> None:
        """Initialize the model adapter.

        Args:
            config: Configuration object containing model settings
            plugins: List of plugin instances to use for lifecycle hooks
        """
        self.config = config
        self.plugins: list[PluginBase] = plugins or []

        logger.info(f"Initialized {self.name} adapter")
        if self.plugins:
            logger.info(f"Loaded {len(self.plugins)} plugins: {[p.name for p in self.plugins]}")

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory.

        This method should:
        1. Check if model is already loaded (skip if so)
        2. Download model from source if not cached
        3. Load model into memory with appropriate dtype
        4. Move model to target device (CUDA/CPU)
        5. Apply model-specific optimizations

        Raises
        ------
        Exception
            If model loading fails (network issues, CUDA errors, etc.)
        """
        pass

    @abstractmethod
    def generate(self, **kwargs) -> Image.Image:
        """Generate or edit an image.

        The specific parameters depend on the model type:
        - text-to-image: prompt, width, height, steps, seed, guidance_scale
        - image-edit: input_image, instruction, seed
        - img2img: input_image, prompt, strength, seed
        - inpainting: input_image, mask, prompt, seed

        Returns
        -------
        Image.Image
            Generated or edited PIL Image

        Raises
        ------
        Exception
            If generation fails
        """
        pass

    @abstractmethod
    def generate_and_save(
        self, output_path: Path | None = None, **kwargs
    ) -> tuple[Image.Image, Path]:
        """Generate an image and save it to disk with plugin hooks.

        This method orchestrates the full generation pipeline:
        1. Call on_generate_start plugin hooks (can modify params)
        2. Generate image using potentially modified params
        3. Call on_generate_complete plugin hooks (can modify image)
        4. Determine output path (auto-generate if not provided)
        5. Call on_before_save plugin hooks (can modify image/path)
        6. Save image to disk
        7. Call on_after_save plugin hooks (e.g., metadata export)

        Args:
            output_path: Custom output path (if None, auto-generates in outputs_dir)
            **kwargs: Model-specific generation parameters

        Returns
        -------
        tuple[Image.Image, Path]
            Tuple of (generated image, save path)

        Raises
        ------
        Exception
            If generation or save fails
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory.

        This method should:
        1. Delete model instance
        2. Clear CUDA cache if using GPU
        3. Reset model loaded flag
        4. Log unload success
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory.

        Returns
        -------
        bool
            True if model is loaded, False otherwise
        """
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Get information about this model adapter.

        Returns
        -------
        dict[str, Any]
            Dictionary containing model metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "version": self.version,
            "is_loaded": self.is_loaded,
        }


class ModelRegistry:
    """Registry for managing available model adapters.

    The registry provides a central location for discovering and instantiating
    model adapters. It follows the same pattern as PluginRegistry and
    WorkflowRegistry for consistency.

    The registry maintains:
    - Registered adapter classes (templates)
    - Instantiated adapter instances (active models)
    - Model metadata for discovery

    Usage
    -----
    Registering a new adapter:

        >>> from pipeworks.core.model_adapters import model_registry
        >>> model_registry.register(MyCustomAdapter)

    Instantiating an adapter:

        >>> adapter = model_registry.instantiate("Z-Image-Turbo", config)
        >>> image = adapter.generate(prompt="test", seed=42)

    Listing available models:

        >>> models = model_registry.list_available()
        >>> for model in models:
        ...     info = model_registry.get_adapter_info(model)
        ...     print(f"{info['name']}: {info['description']}")

    Notes
    -----
    - Adapters must be registered before they can be instantiated
    - Multiple instances of the same adapter can exist
    - Registry is global and shared across the application
    """

    def __init__(self) -> None:
        """Initialize the model registry."""
        self._adapters: dict[str, type[ModelAdapterBase]] = {}
        self._instances: dict[str, list[ModelAdapterBase]] = {}

    def register(self, adapter_class: type[ModelAdapterBase]) -> None:
        """Register a model adapter class.

        Args:
            adapter_class: Model adapter class to register

        Raises
        ------
        ValueError
            If an adapter with the same name is already registered
        """
        adapter_name = adapter_class.name

        if adapter_name in self._adapters:
            logger.warning(f"Model adapter '{adapter_name}' is already registered, overwriting")

        self._adapters[adapter_name] = adapter_class
        self._instances[adapter_name] = []
        logger.info(f"Registered model adapter: {adapter_name}")

    def instantiate(
        self,
        adapter_name: str,
        config: PipeworksConfig,
        plugins: list[PluginBase] | None = None,
    ) -> ModelAdapterBase:
        """Create an instance of a registered model adapter.

        Args:
            adapter_name: Name of the adapter to instantiate
            config: Configuration object
            plugins: List of plugin instances

        Returns
        -------
        ModelAdapterBase
            New instance of the specified adapter

        Raises
        ------
        KeyError
            If adapter_name is not registered
        """
        if adapter_name not in self._adapters:
            available = ", ".join(self.list_available())
            raise KeyError(
                f"Model adapter '{adapter_name}' not found. " f"Available adapters: {available}"
            )

        adapter_class = self._adapters[adapter_name]
        instance = adapter_class(config=config, plugins=plugins)
        self._instances[adapter_name].append(instance)

        logger.info(f"Instantiated model adapter: {adapter_name}")
        return instance

    def get_adapter_class(self, adapter_name: str) -> type[ModelAdapterBase] | None:
        """Get the adapter class for a given name.

        Args:
            adapter_name: Name of the adapter

        Returns
        -------
        type[ModelAdapterBase] | None
            Adapter class or None if not found
        """
        return self._adapters.get(adapter_name)

    def list_available(self) -> list[str]:
        """List all registered adapter names.

        Returns
        -------
        list[str]
            List of adapter names
        """
        return list(self._adapters.keys())

    def get_adapter_info(self, adapter_name: str) -> dict[str, Any] | None:
        """Get information about a registered adapter.

        Args:
            adapter_name: Name of the adapter

        Returns
        -------
        dict[str, Any] | None
            Adapter metadata or None if not found
        """
        if adapter_name not in self._adapters:
            return None

        adapter_class = self._adapters[adapter_name]
        return {
            "name": adapter_class.name,
            "description": adapter_class.description,
            "model_type": adapter_class.model_type,
            "version": adapter_class.version,
        }

    def get_adapters_by_type(self, model_type: str) -> list[dict[str, Any]]:
        """Get all adapters that support a specific model type.

        Args:
            model_type: Type of model to filter by

        Returns
        -------
        list[dict[str, Any]]
            List of adapter metadata dictionaries
        """
        return [
            self.get_adapter_info(name)
            for name, adapter_class in self._adapters.items()
            if adapter_class.model_type == model_type
        ]


# Global model registry instance
model_registry = ModelRegistry()
