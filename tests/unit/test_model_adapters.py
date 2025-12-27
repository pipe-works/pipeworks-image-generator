"""Unit tests for model adapters.

This module contains unit tests for the model adapter classes that wrap
HuggingFace Diffusers pipelines. All tests use mocks to avoid downloading
large ML models, keeping CI fast and reliable.

IMPORTANT: Mock classes must stay in sync with HuggingFace pipeline interfaces.
If diffusers updates ZImagePipeline or QwenImageEditPlusPipeline APIs, update:
- MockZImagePipeline.__init__ and __call__ signatures
- MockQwenImageEditPipeline.__init__ and __call__ signatures
Last verified against: diffusers==0.32.1

Testing Strategy
----------------
- Unit tests (this file): Fast, mocked, run in CI
- Integration tests: Slow, real models, skip in CI (marked with @pytest.mark.requires_model)

Coverage Goals
--------------
- ZImageTurboAdapter: 18.75% → 90%
- QwenImageEditAdapter: 10.08% → 80%
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from pipeworks.core.adapters.qwen_image_edit import QwenImageEditAdapter
from pipeworks.core.adapters.zimage_turbo import ZImageTurboAdapter
from pipeworks.core.config import PipeworksConfig

# ============================================================================
# Mock Classes
# ============================================================================


class MockZImagePipeline:
    """Mock for ZImagePipeline that simulates the interface without loading models.

    This mock class mimics the behavior of diffusers.ZImagePipeline, storing
    call parameters for verification in tests while returning realistic outputs.

    Attributes
    ----------
    transformer : MagicMock
        Mock transformer for compilation tests
    device : str | None
        Device the pipeline was moved to
    _called_with : dict
        Dictionary storing all method call parameters for verification
    """

    def __init__(self, *args, **kwargs):
        """Initialize the mock pipeline."""
        self.transformer = MagicMock()
        self.device = None
        self._called_with = {}

    def to(self, device):
        """Mock the to() method for device placement.

        Args:
            device: Target device (e.g., "cuda", "cpu")

        Returns
        -------
        MockZImagePipeline
            Self for method chaining
        """
        self.device = device
        return self

    def enable_model_cpu_offload(self):
        """Mock CPU offload enabling."""
        self._called_with["cpu_offload"] = True

    def enable_attention_slicing(self):
        """Mock attention slicing enabling."""
        self._called_with["attention_slicing"] = True

    def __call__(self, prompt, height, width, num_inference_steps, guidance_scale, generator=None):
        """Mock the pipeline call for image generation.

        Args:
            prompt: Text prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of steps
            guidance_scale: Guidance scale
            generator: Optional torch Generator

        Returns
        -------
        MagicMock
            Mock output with images attribute containing a PIL Image
        """
        # Store call parameters for verification
        self._called_with.update(
            {
                "prompt": prompt,
                "height": height,
                "width": width,
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "generator": generator,
            }
        )

        # Return mock output with same structure as real pipeline
        mock_output = MagicMock()
        mock_output.images = [Image.new("RGB", (width, height))]
        return mock_output


class MockQwenImageEditPipeline:
    """Mock for QwenImageEditPlusPipeline that simulates the interface without loading models.

    This mock class mimics the behavior of diffusers.QwenImageEditPlusPipeline,
    storing call parameters for verification in tests while returning realistic outputs.

    Attributes
    ----------
    transformer : MagicMock
        Mock transformer for compilation tests
    device : str | None
        Device the pipeline was moved to
    _called_with : dict
        Dictionary storing all method call parameters for verification
    """

    def __init__(self, *args, **kwargs):
        """Initialize the mock pipeline."""
        self.transformer = MagicMock()
        self.device = None
        self._called_with = {}

    def to(self, device):
        """Mock the to() method for device placement.

        Args:
            device: Target device (e.g., "cuda", "cpu")

        Returns
        -------
        MockQwenImageEditPipeline
            Self for method chaining
        """
        self.device = device
        return self

    def enable_sequential_cpu_offload(self):
        """Mock sequential CPU offload enabling."""
        self._called_with["sequential_offload"] = True

    def enable_model_cpu_offload(self):
        """Mock model CPU offload enabling."""
        self._called_with["model_offload"] = True

    def enable_attention_slicing(self):
        """Mock attention slicing enabling."""
        self._called_with["attention_slicing"] = True

    def enable_xformers_memory_efficient_attention(self):
        """Mock xformers attention enabling."""
        self._called_with["xformers"] = True

    def __call__(
        self,
        image,
        prompt,
        num_inference_steps,
        guidance_scale,
        true_cfg_scale,
        negative_prompt,
        generator=None,
        num_images_per_prompt=1,
    ):
        """Mock the pipeline call for image editing.

        Args:
            image: Input image or list of images
            prompt: Editing instruction
            num_inference_steps: Number of steps
            guidance_scale: Guidance scale
            true_cfg_scale: True CFG scale
            negative_prompt: Negative prompt
            generator: Optional torch Generator
            num_images_per_prompt: Number of images to generate

        Returns
        -------
        MagicMock
            Mock output with images attribute containing edited PIL Images
        """
        self._called_with.update(
            {
                "image": image,
                "prompt": prompt,
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "true_cfg": true_cfg_scale,
                "neg_prompt": negative_prompt,
                "generator": generator,
            }
        )

        # Return mock output
        mock_output = MagicMock()
        output_size = image[0].size if isinstance(image, list) else image.size
        mock_output.images = [Image.new("RGB", output_size)]
        return mock_output


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_shared_state():
    """Clean up class-level shared state before and after each test.

    This fixture ensures test isolation by resetting the shared model state
    that was introduced to prevent OOM on browser refresh.
    """
    # Clean up before test
    ZImageTurboAdapter._shared_pipe = None
    ZImageTurboAdapter._shared_model_id = None
    ZImageTurboAdapter._instance_count = 0

    yield

    # Clean up after test
    ZImageTurboAdapter._shared_pipe = None
    ZImageTurboAdapter._shared_model_id = None
    ZImageTurboAdapter._instance_count = 0


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration with temporary paths.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns
    -------
    PipeworksConfig
        Configuration object for testing
    """
    return PipeworksConfig(
        device="cpu",
        torch_dtype="float32",
        zimage_model_id="mock/zimage-turbo",
        qwen_model_id="mock/qwen-image-edit",
        outputs_dir=tmp_path / "outputs",
        models_dir=tmp_path / "models",
        compile_model=False,
        enable_model_cpu_offload=False,
        enable_attention_slicing=False,
    )


@pytest.fixture
def zimage_adapter(test_config):
    """Create a ZImageTurboAdapter with test config.

    Args:
        test_config: Test configuration fixture

    Returns
    -------
    ZImageTurboAdapter
        Adapter instance for testing
    """
    return ZImageTurboAdapter(test_config)


@pytest.fixture
def qwen_adapter(test_config):
    """Create a QwenImageEditAdapter with test config.

    Args:
        test_config: Test configuration fixture

    Returns
    -------
    QwenImageEditAdapter
        Adapter instance for testing
    """
    return QwenImageEditAdapter(test_config)


# ============================================================================
# ZImageTurboAdapter Tests - Priority 1: Model Lifecycle
# ============================================================================


@pytest.mark.unit
class TestZImageTurboAdapterLifecycle:
    """Unit tests for ZImageTurboAdapter model lifecycle."""

    @patch("diffusers.ZImagePipeline")
    def test_load_model_success(self, mock_pipeline_class, zimage_adapter):
        """Test successful model loading."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        zimage_adapter.load_model()

        assert zimage_adapter.is_loaded
        assert ZImageTurboAdapter._shared_pipe is not None
        mock_pipeline_class.from_pretrained.assert_called_once()

    @patch("diffusers.ZImagePipeline")
    def test_load_model_with_bfloat16(self, mock_pipeline_class, test_config):
        """Test model loading with bfloat16 dtype."""
        test_config.torch_dtype = "bfloat16"
        adapter = ZImageTurboAdapter(test_config)
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        adapter.load_model()

        call_kwargs = mock_pipeline_class.from_pretrained.call_args.kwargs
        assert call_kwargs["torch_dtype"] == torch.bfloat16

    @patch("diffusers.ZImagePipeline")
    def test_load_model_with_float16(self, mock_pipeline_class, test_config):
        """Test model loading with float16 dtype."""
        test_config.torch_dtype = "float16"
        adapter = ZImageTurboAdapter(test_config)
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        adapter.load_model()

        call_kwargs = mock_pipeline_class.from_pretrained.call_args.kwargs
        assert call_kwargs["torch_dtype"] == torch.float16

    @patch("diffusers.ZImagePipeline")
    def test_load_model_with_float32(self, mock_pipeline_class, test_config):
        """Test model loading with float32 dtype."""
        test_config.torch_dtype = "float32"
        adapter = ZImageTurboAdapter(test_config)
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        adapter.load_model()

        call_kwargs = mock_pipeline_class.from_pretrained.call_args.kwargs
        assert call_kwargs["torch_dtype"] == torch.float32

    @patch("diffusers.ZImagePipeline")
    def test_load_model_already_loaded_skips(self, mock_pipeline_class, zimage_adapter):
        """Test that loading an already-loaded model is skipped."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        zimage_adapter.load_model()
        zimage_adapter.load_model()  # Second call should skip

        # Should only be called once
        mock_pipeline_class.from_pretrained.assert_called_once()

    @patch("diffusers.ZImagePipeline")
    def test_load_model_failure_not_marked_loaded(self, mock_pipeline_class, zimage_adapter):
        """Test that failed loading doesn't mark model as loaded."""
        mock_pipeline_class.from_pretrained.side_effect = RuntimeError("Model not found")

        with pytest.raises(RuntimeError, match="Model not found"):
            zimage_adapter.load_model()

        assert not zimage_adapter.is_loaded

    @patch("diffusers.ZImagePipeline")
    def test_unload_model(self, mock_pipeline_class, zimage_adapter):
        """Test model unloading."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()
        zimage_adapter.load_model()

        zimage_adapter.unload_model()

        assert not zimage_adapter.is_loaded
        assert ZImageTurboAdapter._shared_pipe is None

    def test_unload_model_when_not_loaded_safe(self, zimage_adapter):
        """Test that unloading when not loaded is safe."""
        zimage_adapter.unload_model()  # Should not raise
        assert not zimage_adapter.is_loaded

    @patch("diffusers.ZImagePipeline")
    def test_is_loaded_property(self, mock_pipeline_class, zimage_adapter):
        """Test is_loaded property reflects model state."""
        # Initially not loaded
        assert not zimage_adapter.is_loaded

        # Load model
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()
        zimage_adapter.load_model()
        assert zimage_adapter.is_loaded

        # Unload model
        zimage_adapter.unload_model()
        assert not zimage_adapter.is_loaded

    @patch("diffusers.ZImagePipeline")
    def test_load_model_uses_cache_dir(self, mock_pipeline_class, zimage_adapter):
        """Test that model loading uses the configured cache directory."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        zimage_adapter.load_model()

        call_kwargs = mock_pipeline_class.from_pretrained.call_args.kwargs
        assert "cache_dir" in call_kwargs
        assert str(zimage_adapter.config.models_dir) == call_kwargs["cache_dir"]


# ============================================================================
# ZImageTurboAdapter Tests - Priority 2: Generation
# ============================================================================


@pytest.mark.unit
class TestZImageTurboAdapterGeneration:
    """Unit tests for image generation."""

    @patch("diffusers.ZImagePipeline")
    def test_generate_returns_pil_image(self, mock_pipeline_class, zimage_adapter):
        """Test that generate returns a PIL Image."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()
        zimage_adapter.load_model()

        image = zimage_adapter.generate(prompt="test", seed=42)

        assert isinstance(image, Image.Image)
        assert image.size == (1024, 1024)  # Default size from config

    @patch("diffusers.ZImagePipeline")
    def test_generate_with_custom_dimensions(self, mock_pipeline_class, zimage_adapter):
        """Test generation with custom dimensions."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()
        zimage_adapter.load_model()

        image = zimage_adapter.generate(prompt="test", width=768, height=512, seed=42)

        assert image.size == (768, 512)

    @patch("diffusers.ZImagePipeline")
    def test_guidance_scale_forced_to_zero(self, mock_pipeline_class, zimage_adapter, caplog):
        """Test that guidance_scale is forced to 0.0 for Turbo."""
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        zimage_adapter.load_model()

        with caplog.at_level(logging.WARNING):
            zimage_adapter.generate(prompt="test", guidance_scale=7.5)

        # Check that warning was logged
        assert "must be 0.0 for Turbo" in caplog.text
        # Verify pipeline was called with 0.0
        assert mock_pipeline._called_with["guidance"] == 0.0

    @patch("diffusers.ZImagePipeline")
    @patch("torch.Generator")
    def test_generator_created_with_seed(
        self, mock_generator_class, mock_pipeline_class, zimage_adapter
    ):
        """Test that torch.Generator is created with the provided seed."""
        mock_generator_instance = MagicMock()
        mock_generator_class.return_value = mock_generator_instance
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()
        zimage_adapter.load_model()

        zimage_adapter.generate(prompt="test", seed=42)

        mock_generator_class.assert_called_once_with("cpu")
        mock_generator_instance.manual_seed.assert_called_once_with(42)

    @patch("diffusers.ZImagePipeline")
    def test_generate_auto_loads_model(self, mock_pipeline_class, zimage_adapter):
        """Test that generate auto-loads model if not loaded."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        # Don't call load_model() explicitly
        image = zimage_adapter.generate(prompt="test", seed=42)

        assert zimage_adapter.is_loaded
        assert isinstance(image, Image.Image)

    @patch("diffusers.ZImagePipeline")
    def test_generate_uses_config_defaults(self, mock_pipeline_class, zimage_adapter):
        """Test that generate uses config defaults when params not provided."""
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        zimage_adapter.load_model()

        zimage_adapter.generate(prompt="test")

        # Should use defaults from config
        assert mock_pipeline._called_with["width"] == zimage_adapter.config.default_width
        assert mock_pipeline._called_with["height"] == zimage_adapter.config.default_height
        assert mock_pipeline._called_with["steps"] == zimage_adapter.config.num_inference_steps

    @patch("diffusers.ZImagePipeline")
    def test_generate_without_seed_no_generator(self, mock_pipeline_class, zimage_adapter):
        """Test that generate without seed doesn't create a generator."""
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        zimage_adapter.load_model()

        zimage_adapter.generate(prompt="test")

        # Generator should be None when no seed provided
        assert mock_pipeline._called_with["generator"] is None


# ============================================================================
# ZImageTurboAdapter Tests - Priority 3: Plugin Integration
# ============================================================================


@pytest.mark.unit
class TestZImageTurboAdapterPlugins:
    """Unit tests for plugin integration."""

    @patch("diffusers.ZImagePipeline")
    def test_generate_and_save_calls_all_plugin_hooks(
        self, mock_pipeline_class, zimage_adapter, tmp_path
    ):
        """Test that all plugin hooks are called in order."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        # Create mock plugin
        mock_plugin = MagicMock()
        mock_plugin.enabled = True
        mock_plugin.on_generate_start.return_value = {
            "prompt": "test",
            "seed": 42,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "model_id": "mock/zimage-turbo",
            "model_name": "Z-Image-Turbo",
        }
        mock_plugin.on_generate_complete.side_effect = lambda img, p: img
        mock_plugin.on_before_save.side_effect = lambda img, path, p: (img, path)

        zimage_adapter.plugins = [mock_plugin]
        zimage_adapter.load_model()

        image, path = zimage_adapter.generate_and_save(prompt="test", seed=42)

        # Verify all hooks were called
        mock_plugin.on_generate_start.assert_called_once()
        mock_plugin.on_generate_complete.assert_called_once()
        mock_plugin.on_before_save.assert_called_once()
        mock_plugin.on_after_save.assert_called_once()

    @patch("diffusers.ZImagePipeline")
    def test_disabled_plugin_not_called(self, mock_pipeline_class, zimage_adapter):
        """Test that disabled plugins are not called."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        mock_plugin = MagicMock()
        mock_plugin.enabled = False
        zimage_adapter.plugins = [mock_plugin]
        zimage_adapter.load_model()

        zimage_adapter.generate_and_save(prompt="test", seed=42)

        mock_plugin.on_generate_start.assert_not_called()
        mock_plugin.on_generate_complete.assert_not_called()
        mock_plugin.on_before_save.assert_not_called()
        mock_plugin.on_after_save.assert_not_called()

    @patch("diffusers.ZImagePipeline")
    def test_plugin_can_modify_parameters(self, mock_pipeline_class, zimage_adapter):
        """Test that plugins can modify generation parameters."""
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Plugin that modifies prompt
        mock_plugin = MagicMock()
        mock_plugin.enabled = True

        def modify_params(params):
            params = params.copy()
            params["prompt"] = "modified prompt"
            return params

        mock_plugin.on_generate_start.side_effect = modify_params
        mock_plugin.on_generate_complete.side_effect = lambda img, p: img
        mock_plugin.on_before_save.side_effect = lambda img, path, p: (img, path)

        zimage_adapter.plugins = [mock_plugin]
        zimage_adapter.load_model()

        zimage_adapter.generate_and_save(prompt="original", seed=42)

        # Check that modified prompt was used
        assert mock_pipeline._called_with["prompt"] == "modified prompt"

    @patch("diffusers.ZImagePipeline")
    def test_plugin_can_modify_image(self, mock_pipeline_class, zimage_adapter):
        """Test that plugins can modify the generated image."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        # Plugin that modifies image
        modified_image = Image.new("RGB", (512, 512), color="red")
        mock_plugin = MagicMock()
        mock_plugin.enabled = True
        mock_plugin.on_generate_start.side_effect = lambda p: p
        mock_plugin.on_generate_complete.return_value = modified_image
        mock_plugin.on_before_save.side_effect = lambda img, path, p: (img, path)

        zimage_adapter.plugins = [mock_plugin]
        zimage_adapter.load_model()

        image, _ = zimage_adapter.generate_and_save(prompt="test", seed=42)

        # Verify modified image was returned
        assert image == modified_image

    @patch("diffusers.ZImagePipeline")
    def test_plugin_can_modify_save_path(self, mock_pipeline_class, zimage_adapter, tmp_path):
        """Test that plugins can modify the save path."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        # Plugin that modifies save path
        custom_path = tmp_path / "custom_output.png"
        mock_plugin = MagicMock()
        mock_plugin.enabled = True
        mock_plugin.on_generate_start.side_effect = lambda p: p
        mock_plugin.on_generate_complete.side_effect = lambda img, p: img
        mock_plugin.on_before_save.return_value = (
            Image.new("RGB", (1024, 1024)),
            custom_path,
        )

        zimage_adapter.plugins = [mock_plugin]
        zimage_adapter.load_model()

        _, path = zimage_adapter.generate_and_save(prompt="test", seed=42)

        # Verify custom path was used
        assert path == custom_path
        assert path.exists()

    @patch("diffusers.ZImagePipeline")
    def test_multiple_plugins_called_in_order(self, mock_pipeline_class, zimage_adapter):
        """Test that multiple plugins are called in order."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        call_order = []

        # Plugin 1
        plugin1 = MagicMock()
        plugin1.enabled = True
        plugin1.on_generate_start.side_effect = lambda p: (call_order.append("p1_start"), p)[1]
        plugin1.on_generate_complete.side_effect = lambda img, p: (
            call_order.append("p1_complete"),
            img,
        )[1]
        plugin1.on_before_save.side_effect = lambda img, path, p: (
            call_order.append("p1_before_save"),
            (img, path),
        )[1]
        plugin1.on_after_save.side_effect = lambda img, path, p: call_order.append("p1_after_save")

        # Plugin 2
        plugin2 = MagicMock()
        plugin2.enabled = True
        plugin2.on_generate_start.side_effect = lambda p: (call_order.append("p2_start"), p)[1]
        plugin2.on_generate_complete.side_effect = lambda img, p: (
            call_order.append("p2_complete"),
            img,
        )[1]
        plugin2.on_before_save.side_effect = lambda img, path, p: (
            call_order.append("p2_before_save"),
            (img, path),
        )[1]
        plugin2.on_after_save.side_effect = lambda img, path, p: call_order.append("p2_after_save")

        zimage_adapter.plugins = [plugin1, plugin2]
        zimage_adapter.load_model()

        zimage_adapter.generate_and_save(prompt="test", seed=42)

        # Verify correct order
        expected_order = [
            "p1_start",
            "p2_start",
            "p1_complete",
            "p2_complete",
            "p1_before_save",
            "p2_before_save",
            "p1_after_save",
            "p2_after_save",
        ]
        assert call_order == expected_order

    @patch("diffusers.ZImagePipeline")
    def test_generate_and_save_auto_generates_filename(self, mock_pipeline_class, zimage_adapter):
        """Test that generate_and_save auto-generates filename with timestamp and seed."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()
        zimage_adapter.load_model()

        _, path = zimage_adapter.generate_and_save(prompt="test", seed=42)

        # Verify filename format
        assert path.name.startswith("pipeworks_")
        assert "_seed42" in path.name
        assert path.suffix == ".png"
        assert path.exists()

    @patch("diffusers.ZImagePipeline")
    def test_generate_and_save_respects_custom_output_path(
        self, mock_pipeline_class, zimage_adapter, tmp_path
    ):
        """Test that generate_and_save uses custom output path when provided."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()
        zimage_adapter.load_model()

        custom_path = tmp_path / "custom_image.png"
        _, path = zimage_adapter.generate_and_save(prompt="test", seed=42, output_path=custom_path)

        # Verify custom path was used
        assert path == custom_path
        assert path.exists()

    @patch("diffusers.ZImagePipeline")
    def test_generate_and_save_creates_parent_directories(
        self, mock_pipeline_class, zimage_adapter, tmp_path
    ):
        """Test that generate_and_save creates parent directories if they don't exist."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()
        zimage_adapter.load_model()

        nested_path = tmp_path / "subdir1" / "subdir2" / "image.png"
        _, path = zimage_adapter.generate_and_save(prompt="test", seed=42, output_path=nested_path)

        # Verify parent directories were created
        assert path.parent.exists()
        assert path.exists()

    @patch("diffusers.ZImagePipeline")
    def test_plugin_receives_correct_params_dict(self, mock_pipeline_class, zimage_adapter):
        """Test that plugins receive complete params dict with all fields."""
        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        mock_plugin = MagicMock()
        mock_plugin.enabled = True

        captured_params = None

        def capture_params(params):
            nonlocal captured_params
            captured_params = params.copy()
            return params

        mock_plugin.on_generate_start.side_effect = capture_params
        mock_plugin.on_generate_complete.side_effect = lambda img, p: img
        mock_plugin.on_before_save.side_effect = lambda img, path, p: (img, path)

        zimage_adapter.plugins = [mock_plugin]
        zimage_adapter.load_model()

        zimage_adapter.generate_and_save(
            prompt="test prompt", width=768, height=512, num_inference_steps=15, seed=123
        )

        # Verify params dict has all expected fields
        assert captured_params is not None
        assert captured_params["prompt"] == "test prompt"
        assert captured_params["width"] == 768
        assert captured_params["height"] == 512
        assert captured_params["num_inference_steps"] == 15
        assert captured_params["seed"] == 123
        assert captured_params["guidance_scale"] == 0.0
        assert captured_params["model_id"] == "mock/zimage-turbo"
        assert captured_params["model_name"] == "Z-Image-Turbo"


# ============================================================================
# ZImageTurboAdapter Tests - Priority 4: Configuration & Optimization
# ============================================================================


@pytest.mark.unit
class TestZImageTurboAdapterConfiguration:
    """Unit tests for configuration handling."""

    @patch("diffusers.ZImagePipeline")
    def test_cpu_offloading_enabled(self, mock_pipeline_class, test_config):
        """Test that CPU offloading is enabled when configured."""
        test_config.enable_model_cpu_offload = True
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        adapter = ZImageTurboAdapter(test_config)
        adapter.load_model()

        assert mock_pipeline._called_with.get("cpu_offload") is True

    @patch("diffusers.ZImagePipeline")
    def test_cpu_offloading_disabled_uses_device(self, mock_pipeline_class, test_config):
        """Test that when CPU offloading is disabled, model is moved to device."""
        test_config.enable_model_cpu_offload = False
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        adapter = ZImageTurboAdapter(test_config)
        adapter.load_model()

        # Verify .to(device) was called
        assert mock_pipeline.device == test_config.device

    @patch("diffusers.ZImagePipeline")
    def test_attention_slicing_enabled(self, mock_pipeline_class, test_config):
        """Test that attention slicing is enabled when configured."""
        test_config.enable_attention_slicing = True
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        adapter = ZImageTurboAdapter(test_config)
        adapter.load_model()

        assert mock_pipeline._called_with.get("attention_slicing") is True

    @patch("diffusers.ZImagePipeline")
    def test_model_compilation_enabled(self, mock_pipeline_class, test_config):
        """Test that model compilation is triggered when configured."""
        test_config.compile_model = True
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        adapter = ZImageTurboAdapter(test_config)
        adapter.load_model()

        # Verify compile was called on transformer
        mock_pipeline.transformer.compile.assert_called_once()

    @patch("diffusers.ZImagePipeline")
    def test_attention_backend_configuration(self, mock_pipeline_class, test_config):
        """Test that custom attention backend is set when configured."""
        test_config.attention_backend = "flash-attention-2"
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        adapter = ZImageTurboAdapter(test_config)
        adapter.load_model()

        # Verify set_attention_backend was called
        mock_pipeline.transformer.set_attention_backend.assert_called_once_with("flash-attention-2")

    @patch("diffusers.ZImagePipeline")
    def test_default_attention_backend_not_set(self, mock_pipeline_class, test_config):
        """Test that default attention backend is not explicitly set."""
        test_config.attention_backend = "default"
        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        adapter = ZImageTurboAdapter(test_config)
        adapter.load_model()

        # Verify set_attention_backend was NOT called for default backend
        mock_pipeline.transformer.set_attention_backend.assert_not_called()

    @patch("diffusers.ZImagePipeline")
    def test_all_optimizations_enabled(self, mock_pipeline_class, test_config):
        """Test that all optimizations can be enabled together."""
        test_config.enable_model_cpu_offload = True
        test_config.enable_attention_slicing = True
        test_config.compile_model = True
        test_config.attention_backend = "flash-attention-2"

        mock_pipeline = MockZImagePipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        adapter = ZImageTurboAdapter(test_config)
        adapter.load_model()

        # Verify all optimizations were applied
        assert mock_pipeline._called_with.get("cpu_offload") is True
        assert mock_pipeline._called_with.get("attention_slicing") is True
        mock_pipeline.transformer.compile.assert_called_once()
        mock_pipeline.transformer.set_attention_backend.assert_called_once_with("flash-attention-2")

    @patch("diffusers.ZImagePipeline")
    def test_model_id_configuration(self, mock_pipeline_class, test_config):
        """Test that custom model ID is used when configured."""
        custom_model_id = "custom-org/custom-zimage-model"
        test_config.zimage_model_id = custom_model_id

        mock_pipeline_class.from_pretrained.return_value = MockZImagePipeline()

        adapter = ZImageTurboAdapter(test_config)
        adapter.load_model()

        # Verify custom model ID was used
        call_args = mock_pipeline_class.from_pretrained.call_args
        assert call_args[0][0] == custom_model_id


# ============================================================================
# QwenImageEditAdapter Tests - Preprocessing
# ============================================================================


@pytest.mark.unit
class TestQwenImageEditAdapterPreprocessing:
    """Unit tests for image preprocessing."""

    def test_preprocess_image_resizes_large_image(self, qwen_adapter):
        """Test that large images are resized while maintaining aspect ratio."""
        # Create a large test image
        large_img = Image.new("RGB", (2048, 2048))

        processed = qwen_adapter._preprocess_image(large_img, max_size=1024)

        assert max(processed.size) <= 1024
        assert processed.mode == "RGB"

    def test_preprocess_image_maintains_aspect_ratio(self, qwen_adapter):
        """Test that aspect ratio is maintained during resize."""
        # 2:1 aspect ratio image
        img = Image.new("RGB", (1600, 800))

        processed = qwen_adapter._preprocess_image(img, max_size=1024)

        # Should be 1024x512 (maintaining 2:1 ratio)
        assert processed.size == (1024, 512)

    def test_preprocess_image_converts_to_rgb(self, qwen_adapter):
        """Test that non-RGB images are converted to RGB."""
        rgba_img = Image.new("RGBA", (512, 512))

        processed = qwen_adapter._preprocess_image(rgba_img)

        assert processed.mode == "RGB"

    def test_preprocess_image_small_image_unchanged(self, qwen_adapter):
        """Test that small images are not resized."""
        small_img = Image.new("RGB", (512, 512))

        processed = qwen_adapter._preprocess_image(small_img, max_size=1024)

        assert processed.size == (512, 512)

    def test_preprocess_image_handles_grayscale(self, qwen_adapter):
        """Test that grayscale images are converted to RGB."""
        gray_img = Image.new("L", (512, 512))

        processed = qwen_adapter._preprocess_image(gray_img)

        assert processed.mode == "RGB"
        assert processed.size == (512, 512)


# ============================================================================
# QwenImageEditAdapter Tests - Validation
# ============================================================================


@pytest.mark.unit
class TestQwenImageEditAdapterValidation:
    """Unit tests for input validation."""

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_requires_input_image(self, mock_pipeline_class, qwen_adapter):
        """Test that generate raises error without input image."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()

        with pytest.raises(ValueError, match="input_image is required"):
            qwen_adapter.generate(instruction="test edit")

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_requires_instruction(self, mock_pipeline_class, qwen_adapter):
        """Test that generate raises error without instruction."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()
        img = Image.new("RGB", (512, 512))

        with pytest.raises(ValueError, match="instruction is required"):
            qwen_adapter.generate(input_image=img, instruction="")

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_rejects_empty_image_list(self, mock_pipeline_class, qwen_adapter):
        """Test that generate rejects empty image list."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()

        with pytest.raises(ValueError, match="At least one input image"):
            qwen_adapter.generate(input_image=[], instruction="test")

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_rejects_too_many_images(self, mock_pipeline_class, qwen_adapter):
        """Test that generate rejects more than 3 images."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()
        images = [Image.new("RGB", (512, 512)) for _ in range(4)]

        with pytest.raises(ValueError, match="Maximum of 3 input images"):
            qwen_adapter.generate(input_image=images, instruction="test")

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_accepts_single_image(self, mock_pipeline_class, qwen_adapter):
        """Test that generate accepts a single image."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()
        img = Image.new("RGB", (512, 512))

        result = qwen_adapter.generate(input_image=img, instruction="test edit")

        assert isinstance(result, Image.Image)

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_accepts_multiple_images(self, mock_pipeline_class, qwen_adapter):
        """Test that generate accepts multiple images (up to 3)."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()
        images = [Image.new("RGB", (512, 512)) for _ in range(3)]

        result = qwen_adapter.generate(input_image=images, instruction="test composition")

        assert isinstance(result, Image.Image)


# ============================================================================
# QwenImageEditAdapter Tests - Lifecycle
# ============================================================================


@pytest.mark.unit
class TestQwenImageEditAdapterLifecycle:
    """Unit tests for model lifecycle management."""

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_load_model_success(self, mock_pipeline_class, qwen_adapter):
        """Test successful model loading."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()

        qwen_adapter.load_model()

        assert qwen_adapter.is_loaded
        assert qwen_adapter.pipe is not None
        mock_pipeline_class.from_pretrained.assert_called_once()

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_load_model_already_loaded_skips(self, mock_pipeline_class, qwen_adapter):
        """Test that loading an already-loaded model is skipped."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()

        qwen_adapter.load_model()
        qwen_adapter.load_model()  # Second call should skip

        # Should only be called once
        mock_pipeline_class.from_pretrained.assert_called_once()

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_unload_model(self, mock_pipeline_class, qwen_adapter):
        """Test model unloading."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()

        qwen_adapter.unload_model()

        assert not qwen_adapter.is_loaded
        assert qwen_adapter.pipe is None

    def test_unload_model_when_not_loaded_safe(self, qwen_adapter):
        """Test that unloading when not loaded is safe."""
        qwen_adapter.unload_model()  # Should not raise
        assert not qwen_adapter.is_loaded

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_is_loaded_property(self, mock_pipeline_class, qwen_adapter):
        """Test is_loaded property reflects model state."""
        # Initially not loaded
        assert not qwen_adapter.is_loaded

        # Load model
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()
        assert qwen_adapter.is_loaded

        # Unload model
        qwen_adapter.unload_model()
        assert not qwen_adapter.is_loaded


# ============================================================================
# QwenImageEditAdapter Tests - Generation
# ============================================================================


@pytest.mark.unit
class TestQwenImageEditAdapterGeneration:
    """Unit tests for image editing generation."""

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_returns_pil_image(self, mock_pipeline_class, qwen_adapter):
        """Test that generate returns a PIL Image."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()
        img = Image.new("RGB", (512, 512))

        result = qwen_adapter.generate(input_image=img, instruction="test edit", seed=42)

        assert isinstance(result, Image.Image)

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_auto_loads_model(self, mock_pipeline_class, qwen_adapter):
        """Test that generate auto-loads model if not loaded."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        img = Image.new("RGB", (512, 512))

        # Don't call load_model() explicitly
        result = qwen_adapter.generate(input_image=img, instruction="test edit")

        assert qwen_adapter.is_loaded
        assert isinstance(result, Image.Image)

    @patch("diffusers.QwenImageEditPlusPipeline")
    @patch("torch.Generator")
    def test_generator_created_with_seed(
        self, mock_generator_class, mock_pipeline_class, qwen_adapter
    ):
        """Test that torch.Generator is created with the provided seed."""
        mock_generator_instance = MagicMock()
        mock_generator_class.return_value = mock_generator_instance
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()
        img = Image.new("RGB", (512, 512))

        qwen_adapter.generate(input_image=img, instruction="test", seed=42)

        mock_generator_class.assert_called_once_with("cpu")
        mock_generator_instance.manual_seed.assert_called_once_with(42)

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_uses_default_steps(self, mock_pipeline_class, qwen_adapter):
        """Test that generate uses default 40 steps when not specified."""
        mock_pipeline = MockQwenImageEditPipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        qwen_adapter.load_model()
        img = Image.new("RGB", (512, 512))

        qwen_adapter.generate(input_image=img, instruction="test")

        # Should use default of 40 steps
        assert mock_pipeline._called_with["steps"] == 40

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_with_custom_parameters(self, mock_pipeline_class, qwen_adapter):
        """Test generation with custom parameters."""
        mock_pipeline = MockQwenImageEditPipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        qwen_adapter.load_model()
        img = Image.new("RGB", (512, 512))

        qwen_adapter.generate(
            input_image=img,
            instruction="test edit",
            num_inference_steps=30,
            guidance_scale=2.0,
            true_cfg_scale=5.0,
            negative_prompt="bad quality",
        )

        # Verify custom parameters were used
        assert mock_pipeline._called_with["steps"] == 30
        assert mock_pipeline._called_with["guidance"] == 2.0
        assert mock_pipeline._called_with["true_cfg"] == 5.0
        assert mock_pipeline._called_with["neg_prompt"] == "bad quality"

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_preprocesses_images(self, mock_pipeline_class, qwen_adapter):
        """Test that generate preprocesses input images."""
        mock_pipeline = MockQwenImageEditPipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        qwen_adapter.load_model()

        # Create an RGBA image that needs preprocessing
        rgba_img = Image.new("RGBA", (512, 512))

        qwen_adapter.generate(input_image=rgba_img, instruction="test")

        # Verify the pipeline received a list of images
        called_image = mock_pipeline._called_with["image"]
        assert isinstance(called_image, list)
        # The preprocessed image should be RGB
        assert called_image[0].mode == "RGB"

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_with_multiple_images(self, mock_pipeline_class, qwen_adapter):
        """Test generation with multiple input images for composition."""
        mock_pipeline = MockQwenImageEditPipeline()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        qwen_adapter.load_model()

        images = [Image.new("RGB", (512, 512)) for _ in range(3)]

        qwen_adapter.generate(input_image=images, instruction="compose these")

        # Verify pipeline received all 3 images
        called_images = mock_pipeline._called_with["image"]
        assert len(called_images) == 3


# ============================================================================
# QwenImageEditAdapter Tests - Plugin Integration
# ============================================================================


@pytest.mark.unit
class TestQwenImageEditAdapterPlugins:
    """Unit tests for plugin integration."""

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_and_save_calls_all_plugin_hooks(self, mock_pipeline_class, qwen_adapter):
        """Test that all plugin hooks are called in order."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()

        # Create mock plugin
        mock_plugin = MagicMock()
        mock_plugin.enabled = True
        mock_plugin.on_generate_start.side_effect = lambda p: p
        mock_plugin.on_generate_complete.side_effect = lambda img, p: img
        mock_plugin.on_before_save.side_effect = lambda img, path, p: (img, path)

        qwen_adapter.plugins = [mock_plugin]
        qwen_adapter.load_model()

        img = Image.new("RGB", (512, 512))
        _, _ = qwen_adapter.generate_and_save(input_image=img, instruction="test edit")

        # Verify all hooks were called
        mock_plugin.on_generate_start.assert_called_once()
        mock_plugin.on_generate_complete.assert_called_once()
        mock_plugin.on_before_save.assert_called_once()
        mock_plugin.on_after_save.assert_called_once()

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_generate_and_save_auto_generates_filename(self, mock_pipeline_class, qwen_adapter):
        """Test that generate_and_save auto-generates filename with instruction prefix."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()
        qwen_adapter.load_model()

        img = Image.new("RGB", (512, 512))
        _, path = qwen_adapter.generate_and_save(input_image=img, instruction="make sky blue")

        # Verify filename format
        assert path.name.startswith("qwen_edit_")
        assert "make_sky_blue" in path.name
        assert path.suffix == ".png"
        assert path.exists()

    @patch("diffusers.QwenImageEditPlusPipeline")
    def test_plugin_receives_correct_params_dict(self, mock_pipeline_class, qwen_adapter):
        """Test that plugins receive complete params dict with all fields."""
        mock_pipeline_class.from_pretrained.return_value = MockQwenImageEditPipeline()

        mock_plugin = MagicMock()
        mock_plugin.enabled = True

        captured_params = None

        def capture_params(params):
            nonlocal captured_params
            captured_params = params.copy()
            return params

        mock_plugin.on_generate_start.side_effect = capture_params
        mock_plugin.on_generate_complete.side_effect = lambda img, p: img
        mock_plugin.on_before_save.side_effect = lambda img, path, p: (img, path)

        qwen_adapter.plugins = [mock_plugin]
        qwen_adapter.load_model()

        img = Image.new("RGB", (512, 512))
        qwen_adapter.generate_and_save(
            input_image=img,
            instruction="test instruction",
            num_inference_steps=30,
            guidance_scale=2.0,
            true_cfg_scale=5.0,
            seed=123,
        )

        # Verify params dict has all expected fields
        assert captured_params is not None
        assert captured_params["instruction"] == "test instruction"
        assert captured_params["num_inference_steps"] == 30
        assert captured_params["guidance_scale"] == 2.0
        assert captured_params["true_cfg_scale"] == 5.0
        assert captured_params["seed"] == 123
        assert captured_params["model_id"] == "mock/qwen-image-edit"
        assert captured_params["model_name"] == "Qwen-Image-Edit"


# Mark the module as completed
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
