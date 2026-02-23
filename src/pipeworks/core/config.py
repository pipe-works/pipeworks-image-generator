"""Configuration management for Pipe-Works Image Generator.

This module provides centralised configuration management using Pydantic Settings.
All configuration is loaded from environment variables with the ``PIPEWORKS_`` prefix,
allowing easy customisation without code changes.

Environment Variable Loading
-----------------------------
Configuration values are loaded in the following priority order:

1. Environment variables (``PIPEWORKS_*`` prefix)
2. ``.env`` file in the project root
3. Default values defined in :class:`PipeworksConfig`

Example ``.env`` file::

    PIPEWORKS_DEVICE=cuda
    PIPEWORKS_NUM_INFERENCE_STEPS=9
    PIPEWORKS_OUTPUTS_DIR=outputs

Global Configuration Instance
------------------------------
A global :data:`config` instance is created automatically at module import time.
This ensures a single source of truth for all configuration values across the
application.

Usage Example
-------------
::

    from pipeworks.core.config import config

    # Access configuration values
    print(config.device)           # "cuda"
    print(config.outputs_dir)      # Path("outputs")
    print(config.server_port)      # 7860

Directory Management
--------------------
The configuration automatically creates required directories on initialisation:

- ``models_dir``  — Cached HuggingFace model files
- ``outputs_dir`` — Generated images (used by the gallery)
- ``gallery_dir`` — Web-accessible gallery images (inside ``static/``)

Z-Image-Turbo Constraints
--------------------------
Important constraints for Z-Image-Turbo model:

- ``guidance_scale`` MUST be 0.0 (enforced in :mod:`pipeworks.core.model_manager`)
- Recommended ``num_inference_steps``: 9 (results in 8 DiT forwards)
- Optimal ``torch_dtype``: bfloat16 (best quality/performance balance)
- Device: cuda preferred, falls back to cpu

Performance Optimisation Settings
----------------------------------
The config provides several optimisation flags:

- ``enable_attention_slicing``   — Reduces VRAM usage at slight speed cost
- ``enable_model_cpu_offload``   — Enables CPU offloading for memory-constrained setups
- ``compile_model``              — Uses ``torch.compile`` for faster inference (slower first run)
- ``attention_backend``          — Can use Flash-Attention-2 for speedup

See Also
--------
- ``.env.example`` : Template with all available configuration options
- :class:`PipeworksConfig` : Full configuration class documentation
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Resolve the package root directory so that ``static/`` and ``templates/``
# paths default to locations within the installed package, not the current
# working directory.
# ---------------------------------------------------------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent.parent  # src/pipeworks/


class PipeworksConfig(BaseSettings):
    """Main configuration for the Pipe-Works Image Generator.

    Uses Pydantic Settings to load values from environment variables prefixed
    with ``PIPEWORKS_``, with fallback to the defaults defined here.  All
    :class:`~pathlib.Path` fields are resolved on initialisation and their
    directories are created automatically.

    Attributes
    ----------
    General Model Settings
        torch_dtype : Literal["bfloat16", "float16", "float32"]
            Torch dtype for model inference (bfloat16 recommended for
            quality/performance balance).
        device : str
            Device for inference — ``cuda``, ``mps``, or ``cpu``.

    Generation Defaults
        num_inference_steps : int
            Default number of diffusion steps (9 recommended for Turbo).
        guidance_scale : float
            Classifier-free guidance scale.  MUST be 0.0 for Turbo models.
        default_width : int
            Default image width in pixels (512–2048, must be multiple of 64).
        default_height : int
            Default image height in pixels (512–2048, must be multiple of 64).

    Performance Optimisation
        enable_attention_slicing : bool
            Enable attention slicing for lower VRAM usage.
        enable_model_cpu_offload : bool
            Enable sequential CPU offloading for memory-constrained setups.
        compile_model : bool
            Compile model with ``torch.compile`` (slower first run, faster
            subsequent inference).
        attention_backend : Literal["default", "flash", "_flash_3"]
            Attention backend selection.  ``"flash"`` enables Flash-Attention-2.

    Paths
        models_dir : Path
            Directory to cache downloaded HuggingFace models.
        outputs_dir : Path
            Directory to save generated images.
        static_dir : Path
            Root of the web-accessible static files directory.
        data_dir : Path
            Directory containing ``models.json`` and ``prompts.json``.
        gallery_dir : Path
            Directory for gallery image files (inside ``static/``).
        templates_dir : Path
            Directory containing the HTML template(s).

    Server Settings
        server_host : str
            Bind address for the uvicorn ASGI server.
        server_port : int
            Port for the uvicorn ASGI server (1024–65535).

    Notes
    -----
    - All directories are created automatically if they do not exist.
    - Configuration is immutable after initialisation.
    - To modify values, set environment variables and restart the application.

    Examples
    --------
    Using the global instance (recommended)::

        >>> from pipeworks.core.config import config
        >>> config.device
        'cuda'

    Creating a custom instance for testing::

        >>> cfg = PipeworksConfig(device="cpu", server_port=9000)
        >>> cfg.server_port
        9000
    """

    # -- Pydantic Settings configuration ------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PIPEWORKS_",
        case_sensitive=False,
    )

    # -- General model settings (shared across all adapters) ----------------
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description=(
            "Torch dtype for model inference.  bfloat16 is recommended for "
            "the best balance of quality and performance on modern GPUs."
        ),
    )
    device: str = Field(
        default="cuda",
        description=("Device to run inference on.  Supported values: cuda, mps, cpu."),
    )

    # -- Generation defaults ------------------------------------------------
    num_inference_steps: int = Field(
        default=9,
        description=(
            "Default number of diffusion inference steps.  For Z-Image-Turbo "
            "this results in 8 DiT forwards (recommended)."
        ),
        ge=1,
        le=50,
    )
    guidance_scale: float = Field(
        default=0.0,
        description=(
            "Classifier-free guidance scale.  MUST be 0.0 for Turbo models.  "
            "Non-turbo models (e.g. SD v1.5, SDXL) typically use 7.0–7.5."
        ),
    )
    default_width: int = Field(
        default=1024,
        description="Default image width in pixels.",
        ge=512,
        le=2048,
    )
    default_height: int = Field(
        default=1024,
        description="Default image height in pixels.",
        ge=512,
        le=2048,
    )

    # -- Performance optimisation -------------------------------------------
    enable_attention_slicing: bool = Field(
        default=False,
        description=(
            "Enable attention slicing to reduce VRAM usage at the cost of "
            "slightly slower inference."
        ),
    )
    enable_model_cpu_offload: bool = Field(
        default=False,
        description=(
            "Enable sequential CPU offloading.  Moves model layers to CPU "
            "when not in use, reducing peak VRAM but increasing latency."
        ),
    )
    compile_model: bool = Field(
        default=False,
        description=(
            "Compile the model with torch.compile for faster inference.  "
            "The first run will be significantly slower (compilation step)."
        ),
    )
    attention_backend: Literal["default", "flash", "_flash_3"] = Field(
        default="default",
        description=(
            "Attention backend selection.  Set to 'flash' to use "
            "Flash-Attention-2 (requires compatible GPU and library)."
        ),
    )

    # -- Paths --------------------------------------------------------------
    models_dir: Path = Field(
        default=Path("models"),
        description="Directory to cache downloaded HuggingFace model files.",
    )
    outputs_dir: Path = Field(
        default=Path("outputs"),
        description="Directory to save generated images.",
    )
    static_dir: Path = Field(
        default=_PACKAGE_DIR / "static",
        description=(
            "Root of the web-accessible static files directory.  Defaults to "
            "the 'static/' directory inside the installed package."
        ),
    )
    data_dir: Path = Field(
        default=_PACKAGE_DIR / "static" / "data",
        description=(
            "Directory containing models.json and prompts.json configuration "
            "files.  Defaults to 'static/data/' inside the installed package."
        ),
    )
    gallery_dir: Path = Field(
        default=_PACKAGE_DIR / "static" / "gallery",
        description=(
            "Directory for gallery image files.  Located inside static/ so "
            "that images are directly web-accessible via /static/gallery/."
        ),
    )
    templates_dir: Path = Field(
        default=_PACKAGE_DIR / "templates",
        description=(
            "Directory containing HTML template files.  Defaults to the "
            "'templates/' directory inside the installed package."
        ),
    )

    # -- Server settings ----------------------------------------------------
    server_host: str = Field(
        default="0.0.0.0",
        description=(
            "Bind address for the uvicorn ASGI server.  Use '0.0.0.0' to "
            "accept connections from any network interface, or '127.0.0.1' "
            "for localhost-only access."
        ),
    )
    server_port: int = Field(
        default=7860,
        description="Port for the uvicorn ASGI server.",
        ge=1024,
        le=65535,
    )

    def __init__(self, **kwargs):
        """Initialise configuration and create required directories.

        After loading values from environment variables and defaults, this
        method ensures that all required directories exist on disk.  Directory
        creation uses ``parents=True`` (creates parent directories) and
        ``exist_ok=True`` (no error if directory already exists), making it
        safe to call multiple times.

        Args:
            **kwargs: Configuration overrides.  Typically supplied via
                environment variables or explicitly in test code.
        """
        super().__init__(**kwargs)

        # Create directories that must exist before the application can
        # function.  Static/data/templates directories ship with the package
        # and are not created here — only runtime-writable directories.
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Global configuration instance — single source of truth for the application.
# Created at import time so that ``from pipeworks.core.config import config``
# provides immediate access without explicit initialisation.
# ---------------------------------------------------------------------------
config = PipeworksConfig()
