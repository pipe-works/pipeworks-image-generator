"""Configuration management for Pipeworks Image Generator."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipeworksConfig(BaseSettings):
    """Main configuration for Pipeworks Image Generator."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PIPEWORKS_",
        case_sensitive=False,
    )

    # Model settings
    model_id: str = Field(
        default="Tongyi-MAI/Z-Image-Turbo",
        description="HuggingFace model ID for image generation",
    )
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
    outputs_dir: Path = Field(
        default=Path("outputs"),
        description="Directory to save generated images",
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
        super().__init__(**kwargs)
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = PipeworksConfig()
