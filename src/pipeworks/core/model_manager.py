"""Diffusion pipeline lifecycle management for the Pipe-Works Image Generator.

This module provides :class:`ModelManager`, the single point of control for
loading, switching, and invoking HuggingFace diffusers pipelines.  It replaces
the previous ``ModelAdapterBase`` / ``ModelRegistry`` pattern with a simpler,
more direct approach: one manager, one pipeline in memory at a time.

Key Responsibilities
--------------------
- **Lazy model loading** — the pipeline is only loaded when ``generate()`` is
  first called (or ``load_model()`` is called explicitly).
- **Model switching** — when a different model is requested the current
  pipeline is unloaded and CUDA memory is freed before loading the new one.
- **Turbo-model enforcement** — models whose HuggingFace ID contains
  ``"turbo"`` (case-insensitive) have their ``guidance_scale`` forced to 0.0.
- **Deterministic generation** — a seeded ``torch.Generator`` is created for
  every call to ``generate()``, ensuring the same seed always produces the
  same image.
- **Performance optimisation** — attention slicing, sequential CPU offloading,
  and ``torch.compile`` can all be enabled via :class:`PipeworksConfig`.
- **CUDA memory management** — on model switch or unload, the pipeline
  reference is deleted, garbage-collected, and ``torch.cuda.empty_cache()``
  is called.

Usage
-----
::

    from pipeworks.core.config import config
    from pipeworks.core.model_manager import ModelManager

    mgr = ModelManager(config)
    mgr.load_model("Tongyi-MAI/Z-Image-Turbo")

    image = mgr.generate(
        prompt="a goblin workshop",
        width=1024,
        height=1024,
        steps=4,
        guidance_scale=0.0,
        seed=42,
    )

    mgr.unload()

See Also
--------
- :mod:`pipeworks.core.config` — configuration fields that control model
  loading and performance.
- :mod:`pipeworks.api.main` — the FastAPI application that instantiates
  and uses ``ModelManager``.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

from PIL import Image

from pipeworks.core.config import PipeworksConfig

if TYPE_CHECKING:
    pass  # Future type-only imports can go here.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dtype string → torch dtype mapping.
# Kept outside the class so it is defined once and reused.  The strings
# correspond to the allowed values of PipeworksConfig.torch_dtype.
# ---------------------------------------------------------------------------
_DTYPE_MAP: dict | None = None


def _get_dtype_map() -> dict:
    """Return the dtype string → ``torch.dtype`` mapping.

    The mapping is lazily constructed on first call so that ``torch`` is not
    imported at module level.  This keeps import time fast when torch is not
    installed (e.g. during documentation builds).

    Returns:
        Dictionary mapping ``"bfloat16"``, ``"float16"``, and ``"float32"``
        to their corresponding ``torch.dtype`` values.
    """
    global _DTYPE_MAP
    if _DTYPE_MAP is None:
        import torch

        _DTYPE_MAP = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
    return _DTYPE_MAP


class ModelManager:
    """Manages the lifecycle of a single diffusers pipeline.

    At any given time, at most one pipeline is loaded in GPU memory.  When a
    different model is requested the current pipeline is unloaded first.

    Attributes:
        _config (PipeworksConfig):
            Application configuration — device, dtype, paths, and
            performance flags.
        _pipeline:
            The currently loaded diffusers pipeline, or ``None`` when no
            model is loaded.
        _current_model_id (str | None):
            HuggingFace identifier of the currently loaded model, or
            ``None``.
    """

    def __init__(self, config: PipeworksConfig) -> None:
        """Initialise the model manager.

        No model is loaded at this stage — use :meth:`load_model` or rely on
        the lazy-loading behaviour of :meth:`generate`.

        Args:
            config: Application configuration instance.  The manager reads
                ``device``, ``torch_dtype``, ``models_dir``, and all
                performance-related fields from this object.
        """
        self._config = config

        # Pipeline state — initialised to "nothing loaded".
        self._pipeline = None
        self._current_model_id: str | None = None

    # -- Public interface ---------------------------------------------------

    def load_model(self, hf_id: str) -> None:
        """Load a diffusers pipeline by HuggingFace model identifier.

        If the requested model is already loaded this method is a no-op.
        If a *different* model is loaded it is unloaded first (freeing GPU
        memory) before loading the new one.

        The pipeline is loaded with:
        - ``torch_dtype`` from ``config.torch_dtype``
        - ``cache_dir`` from ``config.models_dir``
        - Attention slicing if ``config.enable_attention_slicing`` is True
        - Sequential CPU offloading if ``config.enable_model_cpu_offload``
        - ``torch.compile`` if ``config.compile_model`` is True

        Args:
            hf_id: HuggingFace model identifier, e.g.
                ``"Tongyi-MAI/Z-Image-Turbo"`` or
                ``"stabilityai/stable-diffusion-xl-base-1.0"``.

        Raises:
            RuntimeError: If the model cannot be loaded (network error, out
                of memory, incompatible model format, etc.).
        """
        # --- Short-circuit: same model already loaded ----------------------
        if self._current_model_id == hf_id and self._pipeline is not None:
            logger.info("Model '%s' is already loaded — skipping.", hf_id)
            return

        # --- Unload any previously loaded model ----------------------------
        if self._pipeline is not None:
            logger.info(
                "Switching from '%s' to '%s' — unloading current model.",
                self._current_model_id,
                hf_id,
            )
            self.unload()

        # --- Import torch/diffusers here (lazy) ----------------------------
        import torch
        from diffusers import AutoPipelineForText2Image

        # Resolve the torch dtype from the config string.
        dtype_map = _get_dtype_map()
        torch_dtype = dtype_map.get(self._config.torch_dtype, torch.bfloat16)

        logger.info(
            "Loading model '%s' (dtype=%s, device=%s, cache=%s).",
            hf_id,
            self._config.torch_dtype,
            self._config.device,
            self._config.models_dir,
        )

        try:
            # AutoPipelineForText2Image automatically selects the correct
            # pipeline class based on the model's config.json.
            pipeline = AutoPipelineForText2Image.from_pretrained(
                hf_id,
                torch_dtype=torch_dtype,
                cache_dir=str(self._config.models_dir),
            )

            # --- Move to device or enable CPU offloading -------------------
            if self._config.enable_model_cpu_offload:
                # Sequential CPU offloading: each layer moves to GPU only
                # when needed, then back to CPU.  Drastically reduces peak
                # VRAM but increases latency.
                pipeline.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offloading enabled.")
            else:
                # Standard approach: move entire pipeline to the target device.
                pipeline = pipeline.to(self._config.device)

            # --- Optional performance optimisations ------------------------
            if self._config.enable_attention_slicing:
                # Sliced attention computes attention in steps instead of all
                # at once, trading a small amount of speed for lower VRAM.
                pipeline.enable_attention_slicing()
                logger.info("Attention slicing enabled.")

            if self._config.compile_model:
                # torch.compile optimises the UNet for faster inference.
                # The first call will be slower due to compilation.
                pipeline.unet = torch.compile(
                    pipeline.unet,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                logger.info("Model compiled with torch.compile.")

            # --- Store state -----------------------------------------------
            self._pipeline = pipeline
            self._current_model_id = hf_id
            logger.info("Model '%s' loaded successfully.", hf_id)

        except Exception:
            # If loading fails, ensure we are in a clean state so that a
            # subsequent call to generate() does not try to use a partially
            # loaded pipeline.
            self._pipeline = None
            self._current_model_id = None
            logger.exception("Failed to load model '%s'.", hf_id)
            raise

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        """Generate a single image using the currently loaded pipeline.

        If no model is loaded, this method raises ``RuntimeError``.
        Call :meth:`load_model` first.

        For *turbo* models (detected by the substring ``"turbo"`` in the
        model's HuggingFace ID), ``guidance_scale`` is forced to 0.0
        regardless of the value passed by the caller.  This is a hard
        constraint of turbo-distilled models — non-zero guidance produces
        degraded output.

        Deterministic generation is ensured by creating a fresh
        ``torch.Generator`` seeded with *seed* for every call.

        Args:
            prompt: Text prompt describing the desired image.
            width: Image width in pixels (should be a multiple of 64).
            height: Image height in pixels (should be a multiple of 64).
            steps: Number of diffusion inference steps.
            guidance_scale: Classifier-free guidance scale.  Forced to 0.0
                for turbo models.
            seed: Random seed for reproducible generation.
            negative_prompt: Optional text describing what to avoid in the
                generated image.  Not supported by all models (turbo models
                typically ignore it).

        Returns:
            A PIL :class:`~PIL.Image.Image` of the generated result.

        Raises:
            RuntimeError: If no model is currently loaded.
            torch.cuda.OutOfMemoryError: If there is insufficient GPU memory
                to run the pipeline.
        """
        if self._pipeline is None:
            raise RuntimeError("No model is loaded.  Call load_model(hf_id) before generate().")

        import torch

        # --- Turbo guidance enforcement ------------------------------------
        # Turbo-distilled models (e.g. Z-Image-Turbo, SDXL-Turbo) require
        # guidance_scale=0.0.  Non-zero values produce heavily degraded
        # output because the distillation removes the need for classifier-
        # free guidance.
        if self._current_model_id and "turbo" in self._current_model_id.lower():
            if guidance_scale != 0.0:
                logger.warning(
                    "Turbo model detected ('%s') — forcing guidance_scale " "from %.1f to 0.0.",
                    self._current_model_id,
                    guidance_scale,
                )
                guidance_scale = 0.0

        # --- Deterministic seed --------------------------------------------
        # Create a fresh Generator for each call so that generation is fully
        # reproducible: same seed + same prompt + same params = same image.
        generator = torch.Generator(device=self._config.device).manual_seed(seed)

        logger.info(
            "Generating image: %dx%d, %d steps, guidance=%.1f, seed=%d.",
            width,
            height,
            steps,
            guidance_scale,
            seed,
        )

        # --- Build pipeline kwargs -----------------------------------------
        pipeline_kwargs: dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        # Only pass negative_prompt if the caller provided one.  Some models
        # (especially turbo variants) do not support it and may error.
        if negative_prompt:
            pipeline_kwargs["negative_prompt"] = negative_prompt

        # --- Run the pipeline ----------------------------------------------
        output = self._pipeline(**pipeline_kwargs)

        # diffusers pipelines return an object with an `images` list.
        # We always generate one image at a time (batching is handled at the
        # API layer by looping over seeds).
        image: Image.Image = output.images[0]

        logger.info("Image generated successfully (seed=%d).", seed)
        return image

    def unload(self) -> None:
        """Unload the current model and free GPU memory.

        Performs the following cleanup sequence:

        1. Delete the pipeline reference.
        2. Run Python garbage collection to release any reference cycles.
        3. If CUDA is available, empty the CUDA memory cache and synchronise
           the GPU to ensure all memory is returned to the OS.

        This method is safe to call when no model is loaded (no-op).
        """
        if self._pipeline is None:
            # Nothing to unload.
            return

        model_id = self._current_model_id
        logger.info("Unloading model '%s'.", model_id)

        # Step 1: delete the pipeline reference.
        del self._pipeline
        self._pipeline = None
        self._current_model_id = None

        # Step 2: force garbage collection to break any reference cycles
        # that might keep tensors alive.
        gc.collect()

        # Step 3: free CUDA memory if available.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("CUDA cache cleared after unloading '%s'.", model_id)
        except ImportError:
            # torch not installed — nothing to clean up.
            pass

    # -- Properties ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether a model pipeline is currently loaded in memory."""
        return self._pipeline is not None

    @property
    def current_model_id(self) -> str | None:
        """HuggingFace ID of the currently loaded model, or ``None``."""
        return self._current_model_id
