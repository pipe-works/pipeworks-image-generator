"""Pipeline wrapper for Z-Image-Turbo model."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from diffusers import ZImagePipeline
from PIL import Image

from .config import PipeworksConfig, config as default_config

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Main image generation pipeline wrapper for Z-Image-Turbo."""

    def __init__(self, config: Optional[PipeworksConfig] = None):
        """
        Initialize the image generator.

        Args:
            config: Configuration object. If None, uses global default config.
        """
        self.config = config or default_config
        self.pipe: Optional[ZImagePipeline] = None
        self._model_loaded = False

        logger.info(f"Initialized ImageGenerator with model: {self.config.model_id}")

    def load_model(self) -> None:
        """Load the Z-Image-Turbo model into memory."""
        if self._model_loaded:
            logger.info("Model already loaded, skipping...")
            return

        logger.info(f"Loading model {self.config.model_id}...")

        try:
            # Map dtype string to torch dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map[self.config.torch_dtype]

            # Load pipeline
            self.pipe = ZImagePipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,
                cache_dir=str(self.config.models_dir),
            )

            # Move to device
            if not self.config.enable_model_cpu_offload:
                self.pipe.to(self.config.device)
            else:
                self.pipe.enable_model_cpu_offload()
                logger.info("Enabled model CPU offloading")

            # Apply optimizations
            if self.config.enable_attention_slicing:
                self.pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing")

            if self.config.attention_backend != "default":
                self.pipe.transformer.set_attention_backend(self.config.attention_backend)
                logger.info(f"Set attention backend to: {self.config.attention_backend}")

            if self.config.compile_model:
                logger.info("Compiling model (this may take a while on first run)...")
                self.pipe.transformer.compile()
                logger.info("Model compiled successfully")

            self._model_loaded = True
            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            width: Image width (default from config)
            height: Image height (default from config)
            num_inference_steps: Number of denoising steps (default from config)
            seed: Random seed for reproducibility (None for random)
            guidance_scale: Guidance scale (should be 0.0 for Turbo)

        Returns:
            Generated PIL Image
        """
        if not self._model_loaded:
            self.load_model()

        # Use config defaults if not specified
        width = width or self.config.default_width
        height = height or self.config.default_height
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.config.guidance_scale

        # Validate guidance_scale for Turbo models
        if guidance_scale != 0.0:
            logger.warning(
                f"guidance_scale is {guidance_scale} but should be 0.0 for Turbo models. "
                "Setting to 0.0."
            )
            guidance_scale = 0.0

        logger.info(f"Generating image: {width}x{height}, steps={num_inference_steps}, seed={seed}")
        logger.info(f"Prompt: {prompt}")

        # Create generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(self.config.device).manual_seed(seed)

        try:
            # Generate image
            output = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            image = output.images[0]
            logger.info("Image generated successfully!")
            return image

        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise

    def generate_and_save(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        output_path: Optional[Path] = None,
    ) -> tuple[Image.Image, Path]:
        """
        Generate an image and save it to disk.

        Args:
            prompt: Text description of the image to generate
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            guidance_scale: Guidance scale
            output_path: Custom output path (if None, auto-generates in outputs_dir)

        Returns:
            Tuple of (generated image, save path)
        """
        # Generate image
        image = self.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            guidance_scale=guidance_scale,
        )

        # Generate output filename if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed_suffix = f"_seed{seed}" if seed is not None else ""
            filename = f"pipeworks_{timestamp}{seed_suffix}.png"
            output_path = self.config.outputs_dir / filename

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")

        return image, output_path

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self._model_loaded:
            logger.info("Unloading model...")
            del self.pipe
            self.pipe = None
            self._model_loaded = False

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded successfully")
