"""Configuration management for Pipeworks Image Generator.

This module provides centralized configuration management using Pydantic Settings.
All configuration is loaded from environment variables with the PIPEWORKS_ prefix,
allowing easy customization without code changes.

Environment Variable Loading
-----------------------------
Configuration values are loaded in the following priority order:
1. Environment variables (PIPEWORKS_* prefix)
2. .env file in the project root
3. Default values defined in PipeworksConfig

Example .env file:
    PIPEWORKS_MODEL_ID=Tongyi-MAI/Z-Image-Turbo
    PIPEWORKS_DEVICE=cuda
    PIPEWORKS_NUM_INFERENCE_STEPS=9
    PIPEWORKS_OUTPUTS_DIR=outputs

Global Configuration Instance
------------------------------
A global `config` instance is created automatically at module import time.
This ensures a single source of truth for all configuration values across
the application.

Usage Example
-------------
    from pipeworks.core.config import config

    # Access configuration values
    print(config.model_id)
    print(config.outputs_dir)

    # Configuration is immutable after initialization
    # To change values, set environment variables and restart

Directory Management
--------------------
The configuration automatically creates required directories on initialization:
- models_dir: For cached model files
- inputs_dir: For prompt builder text files
- outputs_dir: For generated images
- catalog_dir: For archived/favorited images

Z-Image-Turbo Constraints
--------------------------
Important constraints for Z-Image-Turbo model:
- guidance_scale MUST be 0.0 (enforced in adapters/zimage_turbo.py)
- Recommended num_inference_steps: 9 (results in 8 DiT forwards)
- Optimal dtype: bfloat16 (best quality/performance balance)
- Device: cuda preferred, falls back to cpu

Performance Optimization Settings
----------------------------------
The config provides several optimization flags:
- enable_attention_slicing: Reduces VRAM usage at slight speed cost
- enable_model_cpu_offload: Enables CPU offloading for memory-constrained setups
- compile_model: Uses torch.compile for faster inference (slower first run)
- attention_backend: Can use Flash-Attention-2 for speedup

See Also
--------
- .env.example: Template with all available configuration options
- PipeworksConfig: Full configuration class documentation
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipeworksConfig(BaseSettings):
    """Main configuration for Pipeworks Image Generator.

    This class uses Pydantic Settings to manage all application configuration.
    Values are loaded from environment variables with the PIPEWORKS_ prefix,
    with fallback to defaults defined here.

    All Path fields are automatically resolved and created if they don't exist.

    Attributes
    ----------
    Model Adapter Settings:
        default_model_adapter : str
            Default model adapter to use (Z-Image-Turbo, Qwen-Image-Edit, etc.)
        zimage_model_id : str
            HuggingFace model ID for Z-Image-Turbo
        qwen_model_id : str
            HuggingFace model ID for Qwen-Image-Edit
        model_id : str
            [LEGACY] HuggingFace model ID for backward compatibility

    General Model Settings:
        torch_dtype : Literal["bfloat16", "float16", "float32"]
            Torch dtype for model inference (bfloat16 recommended)
        device : str
            Device for inference (cuda, mps, or cpu)

    Generation Settings:
        num_inference_steps : int
            Default number of inference steps (9 recommended for Turbo)
        guidance_scale : float
            Guidance scale (MUST be 0.0 for Turbo models)
        default_width : int
            Default image width in pixels (512-2048)
        default_height : int
            Default image height in pixels (512-2048)

    Performance Optimization:
        enable_attention_slicing : bool
            Enable attention slicing for lower VRAM usage
        enable_model_cpu_offload : bool
            Enable CPU offloading for memory-constrained setups
        compile_model : bool
            Compile model for faster inference (slower first run)
        attention_backend : Literal["default", "flash", "_flash_3"]
            Attention backend (flash for Flash-Attention-2)

    Paths:
        models_dir : Path
            Directory to cache downloaded models
        inputs_dir : Path
            Directory for input text files (prompt builder)
        outputs_dir : Path
            Directory to save generated images
        catalog_dir : Path
            Directory for archived/favorited images

    UI Settings:
        gradio_server_name : str
            Server bind address (0.0.0.0 for local network)
        gradio_server_port : int
            Server port (1024-65535)
        gradio_share : bool
            Create public gradio.live link (keep False for local-only)

    Notes
    -----
    - All directories are created automatically if they don't exist
    - Configuration is immutable after initialization
    - To modify config, set environment variables and restart the application
    - See .env.example for a complete list of configuration options
    - Multiple models are now supported through the model adapter system

    Examples
    --------
    Using model adapters (recommended):

        >>> from pipeworks.core.config import config
        >>> from pipeworks.core import model_registry
        >>> adapter = model_registry.instantiate("Z-Image-Turbo", config)

    Create a custom configuration:

        >>> custom_config = PipeworksConfig(
        ...     default_model_adapter="Qwen-Image-Edit",
        ...     device="cpu",
        ...     num_inference_steps=30
        ... )

    Use the global configuration instance:

        >>> from pipeworks.core.config import config
        >>> print(config.default_model_adapter)
        'Z-Image-Turbo'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PIPEWORKS_",
        case_sensitive=False,
    )

    # Model adapter settings
    default_model_adapter: str = Field(
        default="Z-Image-Turbo",
        description="Default model adapter to use (Z-Image-Turbo, Qwen-Image-Edit, etc.)",
    )

    # Z-Image-Turbo model settings
    model_id: str = Field(
        default="Tongyi-MAI/Z-Image-Turbo",
        description="[LEGACY] HuggingFace model ID for image generation",
    )

    # Model-specific HuggingFace IDs
    zimage_model_id: str = Field(
        default="Tongyi-MAI/Z-Image-Turbo",
        description="HuggingFace model ID for Z-Image-Turbo",
    )
    qwen_model_id: str = Field(
        default="Qwen/Qwen-Image-Edit-2509",
        description="HuggingFace model ID for Qwen-Image-Edit (official model with full support)",
    )

    # General model settings (shared across adapters)
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description="Torch dtype for model inference",
    )
    device: str = Field(
        default="cuda",
        description="Device to run inference on (cuda/cpu)",
    )

    # Z-Image-Turbo specific settings
    num_inference_steps: int = Field(
        default=9,
        description="Number of inference steps (results in 8 DiT forwards)",
        ge=1,
        le=50,
    )
    guidance_scale: float = Field(
        default=0.0,
        description="Guidance scale (MUST be 0.0 for Turbo models)",
    )

    # Default generation settings
    default_width: int = Field(default=1024, ge=512, le=2048)
    default_height: int = Field(default=1024, ge=512, le=2048)

    # Performance optimizations
    enable_attention_slicing: bool = Field(
        default=False,
        description="Enable attention slicing for lower VRAM usage",
    )
    enable_model_cpu_offload: bool = Field(
        default=False,
        description="Enable CPU offloading for memory-constrained setups",
    )
    compile_model: bool = Field(
        default=False,
        description="Compile model for faster inference (slower first run)",
    )
    attention_backend: Literal["default", "flash", "_flash_3"] = Field(
        default="default",
        description="Attention backend (flash for Flash-Attention-2)",
    )

    # Paths
    models_dir: Path = Field(
        default=Path("models"),
        description="Directory to cache models",
    )
    inputs_dir: Path = Field(
        default=Path("inputs"),
        description="Directory for input images and assets",
    )
    outputs_dir: Path = Field(
        default=Path("outputs"),
        description="Directory to save generated images",
    )
    catalog_dir: Path = Field(
        default=Path("catalog"),
        description="Directory for cataloged/archived images",
    )

    # UI settings
    gradio_server_name: str = Field(
        default="0.0.0.0",
        description="Server bind address (0.0.0.0 for local network)",
    )
    gradio_server_port: int = Field(
        default=7860,
        description="Server port",
        ge=1024,
        le=65535,
    )
    gradio_share: bool = Field(
        default=False,
        description="Create public gradio.live link (keep False for local-only)",
    )

    def __init__(self, **kwargs):
        """Initialize configuration and create required directories.

        This method is automatically called when creating a PipeworksConfig instance.
        It ensures all required directories exist, creating them if necessary.

        Args:
            **kwargs: Configuration overrides (typically from environment variables)

        Note:
            Directory creation uses parents=True (creates parent directories) and
            exist_ok=True (no error if directory already exists), making this
            safe to call multiple times.
        """
        super().__init__(**kwargs)

        # Ensure all required directories exist
        # Using parents=True creates any missing parent directories
        # Using exist_ok=True prevents errors if directories already exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
# This instance is created automatically when the module is imported and serves
# as the single source of truth for all configuration values across the application.
# It loads values from environment variables (PIPEWORKS_* prefix) and .env file.
config = PipeworksConfig()
