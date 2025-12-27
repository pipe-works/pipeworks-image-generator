"""Z-Image-Turbo model adapter.

This module provides the adapter for the Z-Image-Turbo model, a fast text-to-image
generation model optimized for speed while maintaining quality.

Z-Image-Turbo Specifics
------------------------
The Z-Image-Turbo model has specific requirements:
- **guidance_scale**: Must be 0.0 (automatically enforced)
- **Optimal steps**: 9 inference steps (results in 8 DiT forwards)
- **Recommended dtype**: bfloat16 for best quality/performance
- **Device**: CUDA preferred, falls back to CPU
- **Model size**: ~12GB

Model Optimization Options
---------------------------
The adapter supports several optimizations (configured via PipeworksConfig):
- **Attention Slicing**: Reduces VRAM usage at slight speed cost
- **CPU Offloading**: Moves model layers to CPU when not in use
- **Model Compilation**: Uses torch.compile for faster inference
- **Flash Attention**: Can use Flash-Attention-2 backend for speedup

Usage Example
-------------
Basic generation:

    >>> from pipeworks.core.adapters.zimage_turbo import ZImageTurboAdapter
    >>> from pipeworks.core.config import config
    >>>
    >>> adapter = ZImageTurboAdapter(config)
    >>> image = adapter.generate(
    ...     prompt="a beautiful landscape",
    ...     seed=42
    ... )

With plugins and auto-save:

    >>> from pipeworks.plugins.base import plugin_registry
    >>> metadata_plugin = plugin_registry.instantiate("Save Metadata")
    >>> adapter = ZImageTurboAdapter(config, plugins=[metadata_plugin])
    >>> image, path = adapter.generate_and_save(
    ...     prompt="a beautiful landscape",
    ...     seed=42
    ... )

See Also
--------
- ModelAdapterBase: Base class for all model adapters
- ZImagePipeline: HuggingFace Diffusers pipeline documentation
- PluginBase: Plugin system documentation
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image

from pipeworks.core.config import PipeworksConfig
from pipeworks.core.model_adapters import ModelAdapterBase, model_registry
from pipeworks.plugins.base import PluginBase

if TYPE_CHECKING:
    from diffusers import ZImagePipeline

logger = logging.getLogger(__name__)


class ZImageTurboAdapter(ModelAdapterBase):
    """Model adapter for Z-Image-Turbo text-to-image generation.

    This adapter wraps the HuggingFace Diffusers ZImagePipeline, providing
    optimized text-to-image generation with Z-Image-Turbo. It handles model
    lifecycle, optimization, and plugin integration.

    The adapter implements lazy loading - the model is only loaded into memory
    when generate() or generate_and_save() is first called. This reduces startup
    time and memory usage when the adapter is initialized but not immediately used.

    Attributes
    ----------
    name : str
        Human-readable name ("Z-Image-Turbo")
    description : str
        Brief description of capabilities
    model_type : str
        Always "text-to-image"
    config : PipeworksConfig
        Configuration object containing model and generation settings
    pipe : ZImagePipeline | None
        HuggingFace Diffusers pipeline (None until loaded)
    plugins : list[PluginBase]
        List of active plugin instances

    Notes
    -----
    - Model loading can take 10-30 seconds depending on hardware and network
    - The model cache is stored in config.models_dir (typically ./models/)
    - Model size is approximately 12GB for Z-Image-Turbo
    - First generation after loading takes longer due to CUDA initialization
    - guidance_scale is automatically forced to 0.0 for Turbo compatibility

    Examples
    --------
    Basic usage:

        >>> adapter = ZImageTurboAdapter(config)
        >>> image = adapter.generate(
        ...     prompt="a serene mountain landscape",
        ...     width=1024,
        ...     height=1024,
        ...     num_inference_steps=9,
        ...     seed=42
        ... )
        >>> image.save("output.png")

    With plugins and auto-save:

        >>> from pipeworks.plugins.base import plugin_registry
        >>> plugins = [plugin_registry.instantiate("Save Metadata")]
        >>> adapter = ZImageTurboAdapter(config, plugins=plugins)
        >>> image, path = adapter.generate_and_save(
        ...     prompt="a serene mountain landscape",
        ...     seed=42
        ... )
        >>> print(f"Saved to: {path}")

    Resource cleanup:

        >>> adapter.unload_model()  # Free VRAM/RAM
    """

    name = "Z-Image-Turbo"
    description = "Fast text-to-image generation with Z-Image-Turbo"
    model_type = "text-to-image"
    version = "1.0.0"

    # Class-level shared model state (shared across all instances)
    # This prevents multiple instances from loading the model multiple times
    # which would cause OOM errors on browser refresh
    _shared_pipe: "ZImagePipeline | None" = None
    _shared_model_id: str | None = None
    _instance_count: int = 0
    _load_lock = threading.Lock()

    def __init__(self, config: PipeworksConfig, plugins: list[PluginBase] | None = None) -> None:
        """Initialize the Z-Image-Turbo adapter.

        Args:
            config: Configuration object containing model settings
            plugins: List of plugin instances to use
        """
        super().__init__(config, plugins)

        # Get model ID from config (should be in PIPEWORKS_ZIMAGE_MODEL_ID env var)
        self.model_id = getattr(config, "zimage_model_id", "Tongyi-MAI/Z-Image-Turbo")

        # Increment instance count (for reference tracking)
        with ZImageTurboAdapter._load_lock:
            ZImageTurboAdapter._instance_count += 1
            logger.info(
                f"Configured Z-Image-Turbo with model: {self.model_id} "
                f"(instance #{ZImageTurboAdapter._instance_count})"
            )

    def load_model(self) -> None:
        """Load the Z-Image-Turbo model into memory.

        This method uses a class-level shared pipeline to prevent multiple instances
        from loading the same model multiple times. This is crucial for preventing
        OOM errors when browser refresh creates new sessions.

        The loading process follows these steps:
        1. Acquire lock to prevent race conditions
        2. Check if shared pipeline already exists with the same model ID
        3. If yes, reuse it; if no, load new model into shared pipeline
        4. Apply performance optimizations
        5. Mark model as loaded

        Raises
        ------
        Exception
            If model loading fails (network issues, CUDA errors, etc.)

        Notes
        -----
        - First instance loads the model, subsequent instances reuse it
        - All instances share the same GPU memory
        - Model is only unloaded when all instances are cleaned up
        - Thread-safe for concurrent session creation
        """
        with ZImageTurboAdapter._load_lock:
            # Check if we already have a loaded model with the same model_id
            if (
                ZImageTurboAdapter._shared_pipe is not None
                and ZImageTurboAdapter._shared_model_id == self.model_id
            ):
                logger.info(
                    f"Reusing already-loaded Z-Image-Turbo model {self.model_id} "
                    f"(shared by {ZImageTurboAdapter._instance_count} instances)"
                )
                return

            # Check if a different model is currently loaded
            if (
                ZImageTurboAdapter._shared_pipe is not None
                and ZImageTurboAdapter._shared_model_id != self.model_id
            ):
                logger.warning(
                    f"Different model already loaded ({ZImageTurboAdapter._shared_model_id}). "
                    f"Unloading before loading {self.model_id}"
                )
                self._unload_shared_model()

            logger.info(f"Loading Z-Image-Turbo model {self.model_id}...")

            try:
                # Import diffusers only when actually loading the model (lazy import)
                from diffusers import ZImagePipeline

                # Map dtype string to torch dtype enum
                # bfloat16 is recommended for best quality/performance balance
                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }
                torch_dtype = dtype_map[self.config.torch_dtype]

                # Load pipeline from HuggingFace Hub (or local cache)
                # low_cpu_mem_usage=False ensures faster loading at cost of higher peak RAM
                ZImageTurboAdapter._shared_pipe = ZImagePipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=False,
                    cache_dir=str(self.config.models_dir),
                )

                # Move model to target device (CUDA preferred for speed)
                # If CPU offloading is enabled, model components are moved dynamically
                assert (
                    ZImageTurboAdapter._shared_pipe is not None
                ), "Pipeline should be loaded at this point"

                if not self.config.enable_model_cpu_offload:
                    # Standard approach: keep entire model on device
                    ZImageTurboAdapter._shared_pipe.to(self.config.device)  # type: ignore[union-attr]
                else:
                    # Memory-efficient approach: move layers to CPU when not in use
                    ZImageTurboAdapter._shared_pipe.enable_model_cpu_offload()  # type: ignore[union-attr]
                    logger.info("Enabled model CPU offloading")

                # Apply performance optimizations
                # Attention slicing reduces VRAM usage at slight speed cost
                if self.config.enable_attention_slicing:
                    ZImageTurboAdapter._shared_pipe.enable_attention_slicing()  # type: ignore[union-attr]
                    logger.info("Enabled attention slicing")

                # Use alternative attention backend (e.g., Flash-Attention-2)
                if self.config.attention_backend != "default":
                    ZImageTurboAdapter._shared_pipe.transformer.set_attention_backend(self.config.attention_backend)  # type: ignore[union-attr]
                    logger.info(f"Set attention backend to: {self.config.attention_backend}")

                # Compile model with torch.compile for faster inference
                # First run is slower, subsequent runs are faster
                if self.config.compile_model:
                    logger.info("Compiling model (this may take a while on first run)...")
                    ZImageTurboAdapter._shared_pipe.transformer.compile()  # type: ignore[union-attr]
                    logger.info("Model compiled successfully")

                ZImageTurboAdapter._shared_model_id = self.model_id
                logger.info(
                    f"Z-Image-Turbo model loaded successfully! "
                    f"(shared by {ZImageTurboAdapter._instance_count} instances)"
                )

            except Exception as e:
                logger.error(f"Failed to load Z-Image-Turbo model: {e}")
                ZImageTurboAdapter._shared_pipe = None
                ZImageTurboAdapter._shared_model_id = None
                raise

    def generate(self, **kwargs) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            width: Image width in pixels (default from config)
            height: Image height in pixels (default from config)
            num_inference_steps: Number of denoising steps (default from config)
            seed: Random seed for reproducibility (None for random)
            guidance_scale: Guidance scale (automatically set to 0.0 for Turbo)

        Returns
        -------
        Image.Image
            Generated PIL Image

        Raises
        ------
        Exception
            If generation fails

        Notes
        -----
        - If model is not loaded, it will be loaded automatically
        - guidance_scale is forced to 0.0 for Z-Image-Turbo compatibility
        - Same seed + prompt + params = same image (reproducible)
        """
        if not self.is_loaded:
            self.load_model()

        # Extract parameters from kwargs
        prompt: str = kwargs.get("prompt", "")
        width: int | None = kwargs.get("width")
        height: int | None = kwargs.get("height")
        num_inference_steps: int | None = kwargs.get("num_inference_steps")
        seed: int | None = kwargs.get("seed")
        guidance_scale: float | None = kwargs.get("guidance_scale")

        # Use config defaults if not specified
        width = width or self.config.default_width
        height = height or self.config.default_height
        num_inference_steps = num_inference_steps or self.config.num_inference_steps

        # Force guidance_scale to 0.0 for Turbo models
        # This is a hard constraint of the Turbo architecture
        if guidance_scale is not None and guidance_scale != 0.0:
            logger.warning(
                f"guidance_scale is {guidance_scale} but must be 0.0 for Turbo models. "
                "Forcing to 0.0."
            )
        guidance_scale = 0.0

        logger.info(f"Generating image: {width}x{height}, steps={num_inference_steps}, seed={seed}")
        logger.info(f"Prompt: {prompt}")

        # Create generator for reproducibility
        # When seed is provided, torch.Generator ensures deterministic results
        # Same seed + prompt + params = same image
        generator = None
        if seed is not None:
            generator = torch.Generator(self.config.device).manual_seed(seed)

        try:
            # Generate image using shared pipeline
            assert (
                ZImageTurboAdapter._shared_pipe is not None
            ), "Pipeline should be loaded at this point"
            output = ZImageTurboAdapter._shared_pipe(  # type: ignore[operator]
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            image: Image.Image = output.images[0]  # type: ignore[no-any-return]
            logger.info("Image generated successfully!")
            return image  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise

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
            prompt: Text description of the image to generate
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            guidance_scale: Guidance scale (forced to 0.0)
            output_path: Custom output path (if None, auto-generates in outputs_dir)

        Returns
        -------
        tuple[Image.Image, Path]
            Tuple of (generated image, save path)

        Raises
        ------
        Exception
            If generation or save fails
        """
        # Extract parameters from kwargs
        prompt: str = kwargs.get("prompt", "")
        width: int | None = kwargs.get("width")
        height: int | None = kwargs.get("height")
        num_inference_steps: int | None = kwargs.get("num_inference_steps")
        seed: int | None = kwargs.get("seed")
        # Note: guidance_scale ignored - Turbo models require 0.0

        # Use config defaults if not specified
        width = width or self.config.default_width
        height = height or self.config.default_height
        num_inference_steps = num_inference_steps or self.config.num_inference_steps

        # Build params dict for plugins
        params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "guidance_scale": 0.0,  # Always 0.0 for Turbo
            "model_id": self.model_id,
            "model_name": self.name,
        }

        # Plugin Hook 1: on_generate_start
        # Allows plugins to modify generation parameters before generation
        for plugin in self.plugins:
            if plugin.enabled:
                params = plugin.on_generate_start(params)

        # Generate image using potentially modified params from plugins
        image = self.generate(**params)

        # Plugin Hook 2: on_generate_complete
        # Allows plugins to modify the generated image after generation
        for plugin in self.plugins:
            if plugin.enabled:
                image = plugin.on_generate_complete(image, params)

        # Generate output filename if not provided by caller
        # Format: pipeworks_YYYYMMDD_HHMMSS_seed{seed}.png
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed_suffix = f"_seed{params['seed']}" if params["seed"] is not None else ""
            filename = f"pipeworks_{timestamp}{seed_suffix}.png"
            output_path = self.config.outputs_dir / filename

        # Ensure parent directory exists (handles nested paths)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Plugin Hook 3: on_before_save
        # Allows plugins to modify image or path before saving
        for plugin in self.plugins:
            if plugin.enabled:
                image, output_path = plugin.on_before_save(image, output_path, params)

        # Save image to disk
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")

        # Plugin Hook 4: on_after_save
        # Allows plugins to perform actions after saving
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_after_save(image, output_path, params)

        return image, output_path

    def unload_model(self) -> None:
        """Unload the Z-Image-Turbo model from memory.

        This method uses reference counting to ensure the model is only unloaded
        when no instances are using it. This is important for shared model state.

        Notes
        -----
        - Model is only actually unloaded when instance count reaches 0
        - This prevents unloading while other sessions are using the model
        - Browser refresh scenarios are handled gracefully
        """
        with ZImageTurboAdapter._load_lock:
            if ZImageTurboAdapter._shared_pipe is None:
                logger.debug("No model loaded to unload")
                return

            # Decrement instance count for this unload
            ZImageTurboAdapter._instance_count = max(0, ZImageTurboAdapter._instance_count - 1)

            logger.info(
                f"Unload requested ({ZImageTurboAdapter._instance_count} instances still active)"
            )

            # Only actually unload if no instances are left
            if ZImageTurboAdapter._instance_count == 0:
                self._unload_shared_model()

    @classmethod
    def _unload_shared_model(cls) -> None:
        """Actually unload the shared model from memory.

        This is a class method that handles the physical unloading of the model.
        Should only be called when reference count is 0 or when forcing a reload.
        """
        if cls._shared_pipe is None:
            return

        logger.info("Unloading Z-Image-Turbo shared model...")
        del cls._shared_pipe
        cls._shared_pipe = None
        cls._shared_model_id = None

        # Clear CUDA cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU to finish freeing memory

        logger.info("Z-Image-Turbo model unloaded successfully")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded.

        Returns
        -------
        bool
            True if model is loaded, False otherwise

        Notes
        -----
        This checks the class-level shared pipeline, not instance state.
        """
        return (
            ZImageTurboAdapter._shared_pipe is not None
            and ZImageTurboAdapter._shared_model_id == self.model_id
        )


# Register the adapter with the global model registry
model_registry.register(ZImageTurboAdapter)
