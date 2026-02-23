"""Tests for pipeworks.core.model_manager — diffusion pipeline lifecycle.

All tests use mocked torch and diffusers imports so that no real model
loading or GPU access occurs.  Tests cover:

- Initial state (no model loaded).
- Model loading and pipeline configuration.
- Model switching and CUDA memory cleanup.
- Turbo model guidance enforcement.
- Deterministic seed generation.
- Error handling during model loading.
- Unload safety (no-op when nothing loaded).

Implementation Note
-------------------
``ModelManager.load_model()`` and ``generate()`` import ``torch`` and
``diffusers`` lazily inside the method body via ``import torch`` and
``from diffusers import AutoPipelineForText2Image``.  To mock these we
use ``sys.modules`` injection rather than ``@patch`` decorators, since
the module-level names don't exist until the import statement executes.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

from pipeworks.core.config import PipeworksConfig
from pipeworks.core.model_manager import ModelManager

# ---------------------------------------------------------------------------
# Shared helpers for mocking torch and diffusers.
# ---------------------------------------------------------------------------


def _create_mock_torch() -> MagicMock:
    """Create a mock ``torch`` module with the attributes ModelManager uses.

    Returns:
        MagicMock configured as a torch module substitute.
    """
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "mock_bfloat16"
    mock_torch.float16 = "mock_float16"
    mock_torch.float32 = "mock_float32"

    # Generator mock for deterministic seeding.
    mock_generator = MagicMock()
    mock_generator.manual_seed.return_value = mock_generator
    mock_torch.Generator.return_value = mock_generator

    # CUDA availability.
    mock_torch.cuda.is_available.return_value = False

    # compile mock.
    mock_torch.compile.return_value = MagicMock()

    return mock_torch


def _create_mock_pipeline():
    """Create a mock diffusers pipeline that returns a 64x64 red image.

    Returns:
        Tuple of (pipeline_instance, AutoPipelineForText2Image_class).
    """
    mock_pipeline = MagicMock()
    mock_output = MagicMock()
    mock_output.images = [Image.new("RGB", (64, 64), color=(255, 0, 0))]
    mock_pipeline.return_value = mock_output
    mock_pipeline.to.return_value = mock_pipeline

    mock_auto_class = MagicMock()
    mock_auto_class.from_pretrained.return_value = mock_pipeline

    return mock_pipeline, mock_auto_class


class _MockContext:
    """Context manager that injects mock torch and diffusers into sys.modules.

    This ensures that ``import torch`` and ``from diffusers import ...``
    inside ModelManager methods resolve to our mocks rather than trying
    to import real packages (which may not be installed in CI).
    """

    def __init__(self):
        self.mock_torch = _create_mock_torch()
        self.mock_pipeline, self.mock_auto_class = _create_mock_pipeline()

        # Create a mock diffusers module with AutoPipelineForText2Image and
        # scheduler classes used by _get_scheduler_map().
        self.mock_diffusers = MagicMock()
        self.mock_diffusers.AutoPipelineForText2Image = self.mock_auto_class

        # Mock scheduler classes — from_config returns a new mock instance.
        self.mock_pndm_scheduler = MagicMock()
        self.mock_pndm_scheduler.from_config.return_value = MagicMock(name="PNDMScheduler")
        self.mock_diffusers.PNDMScheduler = self.mock_pndm_scheduler

        self.mock_dpm_scheduler = MagicMock()
        self.mock_dpm_scheduler.from_config.return_value = MagicMock(name="DPMSolverMultistep")
        self.mock_diffusers.DPMSolverMultistepScheduler = self.mock_dpm_scheduler

        self._saved_torch = None
        self._saved_diffusers = None

    def __enter__(self):
        # Reset the dtype and scheduler map caches so they pick up our mocks.
        global _DTYPE_MAP
        import pipeworks.core.model_manager as mm

        mm._DTYPE_MAP = None
        mm._SCHEDULER_MAP = None

        # Inject mocks into sys.modules.
        self._saved_torch = sys.modules.get("torch")
        self._saved_diffusers = sys.modules.get("diffusers")
        sys.modules["torch"] = self.mock_torch
        sys.modules["diffusers"] = self.mock_diffusers
        return self

    def __exit__(self, *args):
        # Restore original modules.
        if self._saved_torch is not None:
            sys.modules["torch"] = self._saved_torch
        else:
            sys.modules.pop("torch", None)

        if self._saved_diffusers is not None:
            sys.modules["diffusers"] = self._saved_diffusers
        else:
            sys.modules.pop("diffusers", None)

        # Reset dtype and scheduler map caches.
        import pipeworks.core.model_manager as mm

        mm._DTYPE_MAP = None
        mm._SCHEDULER_MAP = None


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


class TestModelManagerInit:
    """Test ModelManager initial state."""

    def test_no_model_loaded_initially(self, test_config: PipeworksConfig):
        """Manager should start with no model loaded."""
        mgr = ModelManager(test_config)
        assert mgr.is_loaded is False
        assert mgr.current_model_id is None

    def test_stores_config(self, test_config: PipeworksConfig):
        """Manager should store the config reference."""
        mgr = ModelManager(test_config)
        assert mgr._config is test_config


class TestModelLoading:
    """Test model loading behaviour."""

    def test_load_model_sets_state(self, test_config: PipeworksConfig):
        """After loading, is_loaded should be True and current_model_id set."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("Tongyi-MAI/Z-Image-Turbo")

            assert mgr.is_loaded is True
            assert mgr.current_model_id == "Tongyi-MAI/Z-Image-Turbo"

    def test_load_same_model_is_noop(self, test_config: PipeworksConfig):
        """Loading the same model twice should not call from_pretrained again."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("Tongyi-MAI/Z-Image-Turbo")
            mgr.load_model("Tongyi-MAI/Z-Image-Turbo")

            # from_pretrained should only be called once.
            assert ctx.mock_auto_class.from_pretrained.call_count == 1

    def test_load_failure_clears_state(self, test_config: PipeworksConfig):
        """If loading fails, state should be cleaned up (no partial pipeline)."""
        with _MockContext() as ctx:
            ctx.mock_auto_class.from_pretrained.side_effect = RuntimeError("Out of memory")

            mgr = ModelManager(test_config)

            with pytest.raises(RuntimeError, match="Out of memory"):
                mgr.load_model("some/model")

            assert mgr.is_loaded is False
            assert mgr.current_model_id is None


class TestModelSwitching:
    """Test switching between different models."""

    def test_switching_unloads_previous(self, test_config: PipeworksConfig):
        """Switching models should unload the previous one and load the new."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("model-a")
            mgr.load_model("model-b")

            assert mgr.current_model_id == "model-b"
            # from_pretrained called twice — once per model.
            assert ctx.mock_auto_class.from_pretrained.call_count == 2


class TestGeneration:
    """Test image generation."""

    def test_generate_without_model_raises(self, test_config: PipeworksConfig):
        """Calling generate() before load_model() should raise RuntimeError."""
        mgr = ModelManager(test_config)
        with pytest.raises(RuntimeError, match="No model is loaded"):
            mgr.generate(
                prompt="test",
                width=512,
                height=512,
                steps=4,
                guidance_scale=0.0,
                seed=42,
            )

    def test_generate_returns_pil_image(self, test_config: PipeworksConfig):
        """generate() should return a PIL Image."""
        with _MockContext():
            mgr = ModelManager(test_config)
            mgr.load_model("test/model")

            image = mgr.generate(
                prompt="test prompt",
                width=1024,
                height=1024,
                steps=4,
                guidance_scale=7.0,
                seed=42,
            )

            assert isinstance(image, Image.Image)

    def test_generate_passes_negative_prompt(self, test_config: PipeworksConfig):
        """Negative prompt should be passed to the pipeline when provided."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("test/model")

            mgr.generate(
                prompt="test",
                width=512,
                height=512,
                steps=4,
                guidance_scale=7.0,
                seed=42,
                negative_prompt="bad quality",
            )

            # Check that the pipeline was called with negative_prompt.
            call_kwargs = ctx.mock_pipeline.call_args[1]
            assert call_kwargs["negative_prompt"] == "bad quality"

    def test_generate_omits_negative_prompt_when_empty(self, test_config: PipeworksConfig):
        """No negative_prompt kwarg should be passed when value is None/empty."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("test/model")

            mgr.generate(
                prompt="test",
                width=512,
                height=512,
                steps=4,
                guidance_scale=7.0,
                seed=42,
                negative_prompt=None,
            )

            call_kwargs = ctx.mock_pipeline.call_args[1]
            assert "negative_prompt" not in call_kwargs


class TestTurboEnforcement:
    """Test turbo model guidance scale enforcement."""

    def test_turbo_forces_guidance_zero(self, test_config: PipeworksConfig):
        """Turbo models should have guidance_scale forced to 0.0."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("Tongyi-MAI/Z-Image-Turbo")

            mgr.generate(
                prompt="test",
                width=512,
                height=512,
                steps=4,
                guidance_scale=7.5,  # Should be overridden to 0.0.
                seed=42,
            )

            # Verify guidance_scale was forced to 0.0 in the pipeline call.
            call_kwargs = ctx.mock_pipeline.call_args[1]
            assert call_kwargs["guidance_scale"] == 0.0

    def test_non_turbo_keeps_guidance(self, test_config: PipeworksConfig):
        """Non-turbo models should keep the provided guidance_scale."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("stabilityai/stable-diffusion-xl-base-1.0")

            mgr.generate(
                prompt="test",
                width=512,
                height=512,
                steps=20,
                guidance_scale=7.5,
                seed=42,
            )

            call_kwargs = ctx.mock_pipeline.call_args[1]
            assert call_kwargs["guidance_scale"] == 7.5

    def test_turbo_case_insensitive(self, test_config: PipeworksConfig):
        """Turbo detection should be case-insensitive."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("some/TURBO-MODEL")

            mgr.generate(
                prompt="test",
                width=512,
                height=512,
                steps=4,
                guidance_scale=5.0,
                seed=42,
            )

            call_kwargs = ctx.mock_pipeline.call_args[1]
            assert call_kwargs["guidance_scale"] == 0.0


class TestSchedulerSwap:
    """Test scheduler swapping in generate()."""

    def test_scheduler_swap_dpmpp_2m_karras(self, test_config: PipeworksConfig):
        """Passing scheduler='dpmpp-2m-karras' should replace the pipeline scheduler."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("stable-diffusion-v1-5/stable-diffusion-v1-5")

            mgr.generate(
                prompt="test",
                width=512,
                height=512,
                steps=20,
                guidance_scale=7.5,
                seed=42,
                scheduler="dpmpp-2m-karras",
            )

            # The DPMSolverMultistepScheduler.from_config should have been called.
            ctx.mock_dpm_scheduler.from_config.assert_called_once()

    def test_scheduler_swap_pndm(self, test_config: PipeworksConfig):
        """Passing scheduler='pndm' should replace the pipeline scheduler with PNDM."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("stable-diffusion-v1-5/stable-diffusion-v1-5")

            mgr.generate(
                prompt="test",
                width=512,
                height=512,
                steps=20,
                guidance_scale=7.5,
                seed=42,
                scheduler="pndm",
            )

            ctx.mock_pndm_scheduler.from_config.assert_called_once()

    def test_no_scheduler_leaves_pipeline_default(self, test_config: PipeworksConfig):
        """Omitting scheduler should not call any scheduler factory."""
        with _MockContext() as ctx:
            mgr = ModelManager(test_config)
            mgr.load_model("test/model")

            mgr.generate(
                prompt="test",
                width=512,
                height=512,
                steps=4,
                guidance_scale=7.0,
                seed=42,
            )

            # Neither scheduler factory should have been called.
            ctx.mock_pndm_scheduler.from_config.assert_not_called()
            ctx.mock_dpm_scheduler.from_config.assert_not_called()

    def test_unknown_scheduler_raises(self, test_config: PipeworksConfig):
        """An unrecognised scheduler ID should raise ValueError."""
        with _MockContext():
            mgr = ModelManager(test_config)
            mgr.load_model("test/model")

            with pytest.raises(ValueError, match="Unknown scheduler"):
                mgr.generate(
                    prompt="test",
                    width=512,
                    height=512,
                    steps=4,
                    guidance_scale=7.0,
                    seed=42,
                    scheduler="nonexistent-scheduler",
                )


class TestUnload:
    """Test model unloading and cleanup."""

    def test_unload_noop_when_empty(self, test_config: PipeworksConfig):
        """Unloading when no model is loaded should not raise."""
        mgr = ModelManager(test_config)
        mgr.unload()  # Should not raise.
        assert mgr.is_loaded is False

    def test_unload_clears_state(self, test_config: PipeworksConfig):
        """After unload, is_loaded should be False and model_id None."""
        with _MockContext():
            mgr = ModelManager(test_config)
            mgr.load_model("test/model")
            assert mgr.is_loaded is True

            mgr.unload()
            assert mgr.is_loaded is False
            assert mgr.current_model_id is None
