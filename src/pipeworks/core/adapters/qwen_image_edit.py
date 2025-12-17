"""Qwen-Image-Edit model adapter.

This module provides the adapter for the Qwen-Image-Edit-2509 model, which performs
instruction-based image editing. Unlike text-to-image models, this takes an existing
image and modifies it based on natural language instructions.

The adapter supports both the official Qwen model and fp8 quantized versions from
aidiffuser repos using a hybrid loading approach that combines the pipeline structure
from the official repo with fp8 weights for reduced memory footprint.

Qwen-Image-Edit Specifics
--------------------------
The Qwen-Image-Edit-2509 model is designed for:
- **Instruction-based editing**: Modify images using natural language
- **Contextual understanding**: Understands complex editing instructions
- **Preservation**: Maintains aspects of the image not mentioned in instruction
- **Multi-modal**: Combines vision and language understanding
- **Multi-image support**: Can composite up to 3 images together

Model Parameters
----------------
Key parameters for this model:
- **image**: Source image(s) to edit (PIL Image or list of PIL Images)
- **prompt**: Natural language editing instruction
- **seed**: Random seed for reproducibility
- **num_inference_steps**: Number of denoising steps (typically 20-50)
- **guidance_scale**: Controls adherence to instruction (typically 1.0)
- **true_cfg_scale**: Controls consistency preservation (typically 4.0)
- **negative_prompt**: Things to avoid in output

Usage Example
-------------
Basic image editing:

    >>> from pipeworks.core.adapters.qwen_image_edit import QwenImageEditAdapter
    >>> from pipeworks.core.config import config
    >>> from PIL import Image
    >>>
    >>> adapter = QwenImageEditAdapter(config)
    >>> base_image = Image.open("character.png")
    >>> edited = adapter.generate(
    ...     input_image=base_image,
    ...     instruction="change the character's hair color to blue",
    ...     seed=42
    ... )

With auto-save:

    >>> edited, path = adapter.generate_and_save(
    ...     input_image=base_image,
    ...     instruction="add a sword to the character's hand",
    ...     seed=42
    ... )

See Also
--------
- ModelAdapterBase: Base class for all model adapters
- ZImageTurboAdapter: Text-to-image generation adapter
- PluginBase: Plugin system documentation
"""

import logging
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

from pipeworks.core.config import PipeworksConfig
from pipeworks.core.model_adapters import ModelAdapterBase, model_registry
from pipeworks.plugins.base import PluginBase

logger = logging.getLogger(__name__)


class QwenImageEditAdapter(ModelAdapterBase):
    """Model adapter for Qwen-Image-Edit-2509 instruction-based image editing.

    This adapter wraps the Qwen-Image-Edit-2509 model pipeline, providing
    instruction-based image editing capabilities. It takes an existing image
    and modifies it based on natural language instructions.

    The adapter implements lazy loading - the model is only loaded into memory
    when generate() or generate_and_save() is first called.

    Attributes
    ----------
    name : str
        Human-readable name ("Qwen-Image-Edit")
    description : str
        Brief description of capabilities
    model_type : str
        Always "image-edit"
    config : PipeworksConfig
        Configuration object containing model and generation settings
    pipe : QwenImageEditPlusPipeline | None
        HuggingFace Diffusers pipeline (None until loaded)
    plugins : list[PluginBase]
        List of active plugin instances

    Notes
    -----
    - Model loading can take 20-40 seconds depending on hardware
    - Model cache is stored in config.models_dir
    - FP8 transformer size: ~20.4GB (fp8_e4m3fn quantized, aidiffuser repos)
    - Full model size: ~57.7GB (bfloat16 precision, official Qwen model)
    - **CPU offloading is automatically enabled for aidiffuser fp8 models**
    - This allows the model to fit in 32GB VRAM by keeping some components in RAM
    - First generation after loading takes longer due to CUDA compilation
    - Supports single or multi-image editing (up to 3 images)
    - Requires VRAM: 12-24GB (fp8 with CPU offload), 40GB+ (full model without offload)

    Examples
    --------
    Basic editing:

        >>> from PIL import Image
        >>> adapter = QwenImageEditAdapter(config)
        >>> img = Image.open("base.png")
        >>> edited = adapter.generate(
        ...     input_image=img,
        ...     instruction="make the sky sunset colors",
        ...     seed=42
        ... )

    With plugins:

        >>> from pipeworks.plugins.base import plugin_registry
        >>> plugins = [plugin_registry.instantiate("Save Metadata")]
        >>> adapter = QwenImageEditAdapter(config, plugins=plugins)
        >>> edited, path = adapter.generate_and_save(
        ...     input_image=img,
        ...     instruction="add dramatic lighting",
        ...     seed=42
        ... )
    """

    name = "Qwen-Image-Edit"
    description = "Instruction-based image editing with Qwen-Image-Edit-2509"
    model_type = "image-edit"
    version = "2.1.0"

    def __init__(
        self, config: PipeworksConfig, plugins: Optional[list[PluginBase]] = None
    ) -> None:
        """Initialize the Qwen-Image-Edit adapter.

        Args:
            config: Configuration object containing model settings
            plugins: List of plugin instances to use

        Notes:
            - Model is not loaded until first generation call
            - Configuration is validated on initialization
        """
        super().__init__(config, plugins)
        self.pipe = None
        self._model_loaded = False

        # Get model ID from config
        self.model_id = getattr(
            config, "qwen_model_id", "Qwen/Qwen-Image-Edit-2509"
        )
        logger.info(f"Configured Qwen-Image-Edit with model: {self.model_id}")

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory cache.

        This should be called before and after inference to ensure
        memory is properly freed between operations.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _preprocess_image(
        self, image: Image.Image, max_size: int = 1024
    ) -> Image.Image:
        """Preprocess image for inference.

        Handles:
        - Resizing large images to prevent OOM
        - Converting to RGB format
        - Ensuring proper dimensions

        Args:
            image: Input PIL Image
            max_size: Maximum dimension size (default 1024)

        Returns:
            Preprocessed PIL Image

        Notes:
            - Images larger than max_size are resized while maintaining aspect ratio
            - Non-RGB images are converted to RGB
            - Resizing uses high-quality LANCZOS resampling
        """
        try:
            # Auto-orient based on EXIF data if available
            try:
                from PIL import ImageOps
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass

            # Resize if too large
            if max(image.size) > max_size:
                original_size = image.size
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.info(
                    f"Resized image from {original_size} to {image.size}"
                )

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
                logger.info(f"Converted image mode to RGB")

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def load_model(self) -> None:
        """Load the Qwen-Image-Edit-2509 model into memory.

        This method downloads the model from HuggingFace (if not cached),
        loads it into memory with the configured dtype, moves it to the
        target device, and applies configured optimizations.

        For fp8 quantized weights (aidiffuser repos), this uses a hybrid approach:
        1. Load pipeline structure from official Qwen/Qwen-Image-Edit-2509
        2. Replace transformer weights with fp8 version from aidiffuser

        The specific QwenImageEditPlusPipeline class is used (not generic
        AutoPipeline) to ensure compatibility with all Qwen-specific features.

        Raises
        ------
        Exception
            If model loading fails (network issues, CUDA errors, etc.)

        Notes
        -----
        - First load downloads ~57.7GB (full) or ~20.4GB (fp8) from HuggingFace
        - Subsequent loads use cache in config.models_dir
        - CUDA compilation on first inference adds ~5-10 seconds
        - Model is moved to GPU unless CPU offload is enabled
        """
        if self._model_loaded:
            logger.info("Qwen-Image-Edit model already loaded, skipping...")
            return

        logger.info(f"Loading Qwen-Image-Edit model {self.model_id}...")
        logger.info(f"Device: {self.config.device}, Dtype: {self.config.torch_dtype}")

        # Suggest memory optimization environment variable
        import os
        if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
            logger.info(
                "TIP: Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                "to reduce CUDA memory fragmentation"
            )

        try:
            # Import the specific pipeline class for Qwen
            from diffusers import QwenImageEditPlusPipeline

            # Map dtype string to torch dtype enum
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

            logger.info(f"Using torch dtype: {torch_dtype}")

            # Check if using aidiffuser fp8 model (only has weight files, no pipeline structure)
            if "aidiffuser" in self.model_id.lower():
                logger.info("Detected aidiffuser repo - using optimized fp8 loading")
                logger.info("Step 1: Loading pipeline configs (no weights) from official Qwen repo...")

                # Load ONLY the pipeline structure/configs from official Qwen (no weights loaded yet)
                official_model_id = "Qwen/Qwen-Image-Edit-2509"

                # First, download just the config files (no model weights)
                from huggingface_hub import snapshot_download
                config_path = snapshot_download(
                    repo_id=official_model_id,
                    cache_dir=str(self.config.models_dir),
                    allow_patterns=["*.json", "*.txt", "*.md"],  # Only download config files
                    ignore_patterns=["*.safetensors", "*.bin", "*.pt"],  # Skip all weight files
                )
                logger.info(f"Downloaded pipeline configs from {official_model_id}")

                # Now download the complete fp8 weights from aidiffuser
                logger.info("Step 2: Downloading complete fp8 model weights from aidiffuser...")
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file

                fp8_weights_path = hf_hub_download(
                    repo_id=self.model_id,
                    filename="Qwen-Image-Edit-2509_fp8_e4m3fn.safetensors",
                    cache_dir=str(self.config.models_dir),
                )
                logger.info(f"Downloaded fp8 weights (20.4GB) to: {fp8_weights_path}")

                # Step 3: Load the complete fp8 weights and build pipeline
                logger.info("Step 3: Loading complete fp8 model weights...")
                fp8_state_dict = load_file(fp8_weights_path, device="cpu")
                logger.info(f"Loaded {len(fp8_state_dict)} keys from fp8 safetensors")

                # Try to use from_single_file if available (cleanest approach)
                try:
                    logger.info("Attempting from_single_file (most efficient method)...")
                    self.pipe = QwenImageEditPlusPipeline.from_single_file(
                        fp8_weights_path,
                        config=config_path,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                    )
                    logger.info("✓ Successfully loaded via from_single_file")
                except (AttributeError, NotImplementedError, TypeError, Exception) as e:
                    logger.warning(f"from_single_file not supported: {type(e).__name__}")
                    logger.error(
                        f"Cannot load aidiffuser fp8 model efficiently. "
                        f"To use Qwen-Image-Edit, please set in your .env file:\n"
                        f"  PIPEWORKS_QWEN_MODEL_ID=Qwen/Qwen-Image-Edit-2509\n"
                        f"Or use the official model which has full pipeline support."
                    )
                    raise RuntimeError(
                        "Aidiffuser fp8 loading requires from_single_file() support. "
                        "Use official Qwen/Qwen-Image-Edit-2509 model instead."
                    )

                # Clean up state dict from memory
                import gc
                del fp8_state_dict
                gc.collect()
            else:
                # Standard loading for official Qwen or other repos with full pipeline structure
                logger.info("Loading from standard diffusers pipeline repo")
                self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    cache_dir=str(self.config.models_dir),
                )

            logger.info("Pipeline loaded successfully (on CPU)")

            # Move model to target device with intelligent memory management
            # Qwen is a large model (~57GB), so we default to CPU offloading for GPUs < 48GB
            use_cpu_offload = self.config.enable_model_cpu_offload

            # Auto-enable CPU offloading for large Qwen models on consumer GPUs
            if not use_cpu_offload and torch.cuda.is_available():
                # Get available VRAM (rough estimate)
                try:
                    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    if total_vram < 48:  # Less than 48GB VRAM
                        logger.info(f"Detected {total_vram:.1f}GB VRAM - enabling CPU offloading for Qwen")
                        use_cpu_offload = True
                except Exception:
                    # If we can't check VRAM, default to CPU offloading for safety
                    logger.warning("Could not check VRAM - defaulting to CPU offloading")
                    use_cpu_offload = True

            if not use_cpu_offload:
                logger.info(f"Moving full model to device: {self.config.device}")
                try:
                    self.pipe.to(self.config.device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"CUDA OOM when moving to GPU: {e}")
                        logger.warning("Falling back to CPU offloading...")
                        use_cpu_offload = True
                    else:
                        raise

            if use_cpu_offload:
                logger.info("Enabling sequential CPU offloading (most aggressive memory saving)")
                logger.info("This reduces VRAM usage significantly but slows inference ~3-5x")
                # Use sequential CPU offload for maximum memory efficiency
                # This moves each layer to GPU only when needed, not the whole component
                try:
                    self.pipe.enable_sequential_cpu_offload()
                    logger.info("Sequential CPU offload enabled successfully")
                except Exception as e:
                    logger.warning(f"Sequential offload failed: {e}, trying model offload...")
                    self.pipe.enable_model_cpu_offload()

            # Apply memory optimizations (always enable for Qwen to reduce VRAM)
            logger.info("Enabling attention slicing to reduce VRAM usage")
            self.pipe.enable_attention_slicing()

            # Apply additional performance optimizations if configured
            if self.config.enable_attention_slicing:
                logger.info("Attention slicing already enabled")

            # Try to enable memory-efficient attention if available
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory-efficient attention")
            except Exception as e:
                logger.debug(f"Could not enable xformers: {e}")

            # Use alternative attention backend if configured
            if self.config.attention_backend != "default":
                try:
                    if hasattr(self.pipe, "transformer"):
                        self.pipe.transformer.set_attention_backend(
                            self.config.attention_backend
                        )
                        logger.info(
                            f"Set attention backend to: {self.config.attention_backend}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not set attention backend: {e}. Continuing with default."
                    )

            self._model_loaded = True
            logger.info("Qwen-Image-Edit model loaded successfully!")

        except ImportError as e:
            logger.error(
                f"Failed to import QwenImageEditPlusPipeline: {e}. "
                "Ensure diffusers is installed: pip install diffusers>=0.28.0"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen-Image-Edit model: {e}")
            raise

    def unload_model(self) -> None:
        """Unload the model from memory.

        This method:
        1. Deletes the pipeline instance
        2. Clears CUDA cache if using GPU
        3. Resets the model loaded flag
        4. Logs the unload operation

        Notes
        -----
        - Safe to call even if model is not loaded
        - Frees all GPU memory used by the model
        - Model can be reloaded by calling load_model() again
        """
        if not self._model_loaded:
            logger.info("Qwen-Image-Edit model not loaded, skipping unload...")
            return

        try:
            logger.info("Unloading Qwen-Image-Edit model...")

            # Delete pipeline instance
            if self.pipe is not None:
                del self.pipe
                self.pipe = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared CUDA cache after model unload")

            # Reset loaded flag
            self._model_loaded = False
            logger.info("Qwen-Image-Edit model unloaded successfully!")

        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            # Don't raise - we want to continue even if unload partially fails
            self._model_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory.

        Returns
        -------
        bool
            True if model is loaded and pipeline exists, False otherwise
        """
        return self._model_loaded and self.pipe is not None

    def generate(
        self,
        input_image: Image.Image | list[Image.Image],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        negative_prompt: str = " ",
    ) -> Image.Image:
        """Edit or composite image(s) based on a natural language instruction.

        Args:
            input_image: Source PIL Image(s) to edit (single image or list of 1-3 images)
            instruction: Natural language instruction for the edit/composition
            num_inference_steps: Number of denoising steps (default 40)
            guidance_scale: How closely to follow instruction (default 1.0)
            true_cfg_scale: Consistency preservation strength (default 4.0)
            seed: Random seed for reproducibility (None for random)
            negative_prompt: Things to avoid in output (default " ")

        Returns
        -------
        Image.Image
            Edited/composited PIL Image

        Raises
        ------
        ValueError
            If input_image is not provided or invalid
        Exception
            If generation fails

        Notes
        -----
        - If model is not loaded, it will be loaded automatically
        - Input images are automatically resized if > 1024px
        - Supports 1-3 input images for composition (e.g., character + hat + background)
        - guidance_scale controls prompt adherence (1.0 is typical)
        - true_cfg_scale controls identity/consistency preservation
        - Same inputs + seed = same output (reproducible)
        - First inference after loading takes longer due to CUDA compilation
        """
        if not self._model_loaded:
            self.load_model()

        if input_image is None:
            raise ValueError("input_image is required for image editing")

        if not instruction or not instruction.strip():
            raise ValueError("instruction is required for image editing")

        # Normalize input_image to always be a list
        if isinstance(input_image, Image.Image):
            images = [input_image]
        else:
            images = input_image

        if len(images) == 0:
            raise ValueError("At least one input image is required")

        if len(images) > 3:
            raise ValueError("Maximum of 3 input images supported")

        # Use reasonable defaults
        num_inference_steps = num_inference_steps or 40

        logger.info(
            f"Editing {len(images)} image(s): steps={num_inference_steps}, "
            f"guidance={guidance_scale}, true_cfg={true_cfg_scale}, seed={seed}"
        )
        logger.info(f"Instruction: {instruction}")

        try:
            # Clear GPU memory before inference
            self._clear_gpu_memory()

            # Preprocess all input images
            processed_images = [self._preprocess_image(img) for img in images]

            logger.info(f"Preprocessed {len(processed_images)} images for editing/composition")

            # Create generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(self.config.device).manual_seed(seed)

            # Measure inference time
            start_time = time.time()

            # Run inference with proper parameters for Qwen-Image-Edit-2509
            # The pipeline expects a list of images
            with torch.inference_mode():
                output = self.pipe(
                    image=processed_images,  # List of 1-3 images
                    prompt=instruction,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    true_cfg_scale=true_cfg_scale,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    num_images_per_prompt=1,
                )

            inference_time = time.time() - start_time

            # Extract output image
            image = output.images[0]

            logger.info(
                f"Image edited successfully in {inference_time:.2f}s. "
                f"Output size: {image.size}"
            )

            return image

        except Exception as e:
            logger.error(f"Failed to edit image: {e}")
            raise

        finally:
            # Always clear GPU memory after inference
            self._clear_gpu_memory()

    def generate_and_save(
        self,
        input_image: Image.Image,
        instruction: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        negative_prompt: str = " ",
        output_path: Optional[Path] = None,
    ) -> tuple[Image.Image, Path]:
        """Edit an image and save it to disk with plugin hooks.

        This method orchestrates the full editing pipeline:
        1. Call on_generate_start plugin hooks (can modify params)
        2. Edit image using potentially modified params
        3. Call on_generate_complete plugin hooks (can modify image)
        4. Determine output path (auto-generate if not provided)
        5. Call on_before_save plugin hooks (can modify image/path)
        6. Save image to disk
        7. Call on_after_save plugin hooks (e.g., metadata export)

        Args:
            input_image: Source PIL Image to edit
            instruction: Natural language instruction for the edit
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow instruction
            true_cfg_scale: Consistency preservation strength
            seed: Random seed for reproducibility
            negative_prompt: Things to avoid in output
            output_path: Custom output path (if None, auto-generates)

        Returns
        -------
        tuple[Image.Image, Path]
            Tuple of (edited image, save path)

        Raises
        ------
        Exception
            If editing or save fails

        Notes
        -----
        - Plugins can modify parameters and output
        - Output path is auto-generated if not provided
        - Auto-generated paths use timestamp and instruction prefix
        - Plugins are called in registration order
        """
        # Call plugin hooks for generation start
        for plugin in self.plugins:
            if hasattr(plugin, "on_generate_start"):
                plugin.on_generate_start(
                    adapter=self,
                    input_image=input_image,
                    instruction=instruction,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )

        # Edit image
        edited_image = self.generate(
            input_image=input_image,
            instruction=instruction,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            negative_prompt=negative_prompt,
        )

        # Call plugin hooks for generation complete
        for plugin in self.plugins:
            if hasattr(plugin, "on_generate_complete"):
                edited_image = plugin.on_generate_complete(
                    adapter=self,
                    image=edited_image,
                    instruction=instruction,
                ) or edited_image

        # Determine output path
        if output_path is None:
            # Auto-generate path with timestamp and instruction prefix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use first 30 chars of instruction, sanitized
            instruction_prefix = (
                instruction[:30]
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )
            filename = f"qwen_edit_{timestamp}_{instruction_prefix}.png"
            output_path = self.config.outputs_dir / filename
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Call plugin hooks for before save
        for plugin in self.plugins:
            if hasattr(plugin, "on_before_save"):
                edited_image, output_path = plugin.on_before_save(
                    adapter=self,
                    image=edited_image,
                    output_path=output_path,
                ) or (edited_image, output_path)

        # Save image
        try:
            edited_image.save(output_path, "PNG")
            logger.info(f"Image saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise

        # Call plugin hooks for after save
        for plugin in self.plugins:
            if hasattr(plugin, "on_after_save"):
                plugin.on_after_save(
                    adapter=self,
                    image=edited_image,
                    output_path=output_path,
                )

        return edited_image, output_path


# Register adapter
model_registry.register(QwenImageEditAdapter)
