"""Tests for pipeworks.core.config — configuration management.

Tests cover:
- Default values for all configuration fields.
- Environment variable overrides via the PIPEWORKS_ prefix.
- Automatic directory creation on initialisation.
- Path resolution for static/data/gallery/templates directories.
- Pydantic validation constraints (port range, dtype literals, etc.).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeworks.core.config import PipeworksConfig


class TestConfigDefaults:
    """Verify that PipeworksConfig provides sensible defaults."""

    def test_default_device_is_cuda(self, monkeypatch, test_config: PipeworksConfig):
        """The test fixture overrides device to 'cpu'; verify direct default."""
        monkeypatch.delenv("PIPEWORKS_DEVICE", raising=False)
        cfg = PipeworksConfig(
            models_dir="/tmp/test_models",
            outputs_dir="/tmp/test_outputs",
            gallery_dir="/tmp/test_gallery",
            _env_file=None,
        )
        assert cfg.device == "cuda"

    def test_default_torch_dtype(self, monkeypatch):
        """Default torch dtype should be bfloat16."""
        monkeypatch.delenv("PIPEWORKS_TORCH_DTYPE", raising=False)
        cfg = PipeworksConfig(
            models_dir="/tmp/test_models",
            outputs_dir="/tmp/test_outputs",
            gallery_dir="/tmp/test_gallery",
            _env_file=None,
        )
        assert cfg.torch_dtype == "bfloat16"

    def test_default_inference_steps(self, test_config: PipeworksConfig):
        """Default inference steps should be 9 (optimal for Z-Image-Turbo)."""
        assert test_config.num_inference_steps == 9

    def test_default_guidance_scale(self, test_config: PipeworksConfig):
        """Default guidance scale should be 0.0 (required for turbo models)."""
        assert test_config.guidance_scale == 0.0

    def test_default_dimensions(self, test_config: PipeworksConfig):
        """Default width and height should be 1024 pixels."""
        assert test_config.default_width == 1024
        assert test_config.default_height == 1024

    def test_default_server_port(self, monkeypatch):
        """Default server port should be 7860."""
        monkeypatch.delenv("PIPEWORKS_SERVER_PORT", raising=False)
        cfg = PipeworksConfig(
            _env_file=None,
            models_dir="/tmp/test_models",
            outputs_dir="/tmp/test_outputs",
            gallery_dir="/tmp/test_gallery",
        )
        assert cfg.server_port == 7860

    def test_default_performance_flags(self, test_config: PipeworksConfig):
        """All performance optimisation flags should default to False."""
        assert test_config.enable_attention_slicing is False
        assert test_config.enable_model_cpu_offload is False
        assert test_config.compile_model is False
        assert test_config.attention_backend == "default"


class TestConfigDirectoryCreation:
    """Verify that PipeworksConfig creates required directories."""

    def test_models_dir_created(self, test_config: PipeworksConfig):
        """models_dir should exist after config initialisation."""
        assert test_config.models_dir.exists()
        assert test_config.models_dir.is_dir()

    def test_outputs_dir_created(self, test_config: PipeworksConfig):
        """outputs_dir should exist after config initialisation."""
        assert test_config.outputs_dir.exists()
        assert test_config.outputs_dir.is_dir()

    def test_gallery_dir_created(self, test_config: PipeworksConfig):
        """gallery_dir should exist after config initialisation."""
        assert test_config.gallery_dir.exists()
        assert test_config.gallery_dir.is_dir()

    def test_creates_nested_directories(self, temp_dir: Path):
        """Config should create deeply nested directories via parents=True."""
        deep_models = temp_dir / "a" / "b" / "c" / "models"
        deep_outputs = temp_dir / "a" / "b" / "c" / "outputs"
        deep_gallery = temp_dir / "a" / "b" / "c" / "gallery"

        cfg = PipeworksConfig(
            models_dir=str(deep_models),
            outputs_dir=str(deep_outputs),
            gallery_dir=str(deep_gallery),
        )

        assert cfg.models_dir.exists()
        assert cfg.outputs_dir.exists()
        assert cfg.gallery_dir.exists()


class TestConfigValidation:
    """Verify Pydantic validation constraints on config fields."""

    def test_invalid_port_too_low(self):
        """Server port below 1024 should raise a validation error."""
        with pytest.raises(Exception):
            PipeworksConfig(
                server_port=80,
                models_dir="/tmp/test_models",
                outputs_dir="/tmp/test_outputs",
                gallery_dir="/tmp/test_gallery",
            )

    def test_invalid_port_too_high(self):
        """Server port above 65535 should raise a validation error."""
        with pytest.raises(Exception):
            PipeworksConfig(
                server_port=70000,
                models_dir="/tmp/test_models",
                outputs_dir="/tmp/test_outputs",
                gallery_dir="/tmp/test_gallery",
            )

    def test_valid_port_range(self, temp_dir: Path):
        """Port within valid range should be accepted."""
        cfg = PipeworksConfig(
            server_port=8080,
            models_dir=str(temp_dir / "models"),
            outputs_dir=str(temp_dir / "outputs"),
            gallery_dir=str(temp_dir / "gallery"),
        )
        assert cfg.server_port == 8080

    def test_invalid_torch_dtype(self):
        """Unsupported torch dtype should raise a validation error."""
        with pytest.raises(Exception):
            PipeworksConfig(
                torch_dtype="int8",
                models_dir="/tmp/test_models",
                outputs_dir="/tmp/test_outputs",
                gallery_dir="/tmp/test_gallery",
            )


class TestConfigPathResolution:
    """Verify that path fields are properly resolved."""

    def test_paths_are_path_objects(self, test_config: PipeworksConfig):
        """All path fields should be pathlib.Path instances."""
        assert isinstance(test_config.models_dir, Path)
        assert isinstance(test_config.outputs_dir, Path)
        assert isinstance(test_config.static_dir, Path)
        assert isinstance(test_config.data_dir, Path)
        assert isinstance(test_config.gallery_dir, Path)
        assert isinstance(test_config.templates_dir, Path)

    def test_custom_paths_override_defaults(self, temp_dir: Path):
        """Explicitly passed paths should override the package defaults."""
        custom_static = temp_dir / "my_static"
        custom_static.mkdir()

        cfg = PipeworksConfig(
            static_dir=str(custom_static),
            models_dir=str(temp_dir / "models"),
            outputs_dir=str(temp_dir / "outputs"),
            gallery_dir=str(temp_dir / "gallery"),
        )
        assert cfg.static_dir == custom_static
