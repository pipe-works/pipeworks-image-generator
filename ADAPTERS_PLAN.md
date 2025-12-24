# Model Adapters Testing Plan

## Executive Summary

This document outlines the strategy to improve test coverage for model adapters in `src/pipeworks/core/adapters/` from the current **~10-18%** to a target of **80-90%** without downloading large ML models in CI.

**Current State:**
- `zimage_turbo.py`: ~18.75% coverage (104 missing lines)
- `qwen_image_edit.py`: ~10.08% coverage (223 missing lines)
- No dedicated unit tests for model adapters
- CI configured with `HF_HUB_OFFLINE=1` (correctly prevents model downloads)

**Goal:**
- Achieve 80-90% coverage using mocking strategies
- Keep CI fast (<5 minutes for adapter tests)
- Maintain separation between unit tests (mocked) and integration tests (real models)

---

## Part 1: Architecture Analysis

### ZImageTurboAdapter (468 lines)

**Model Type:** Text-to-Image
**Base Class:** ModelAdapterBase
**Key Dependencies:** `diffusers.ZImagePipeline`, `torch`, `PIL`

**Critical Paths to Test:**
1. **Model Lifecycle** (lines 155-245)
   - `load_model()`: Model loading with various dtypes and optimizations
   - `unload_model()`: Memory cleanup and CUDA cache clearing
   - `is_loaded` property

2. **Generation Pipeline** (lines 247-327)
   - `generate()`: Image generation with parameter validation
   - Guidance scale enforcement (must be 0.0 for Turbo)
   - Seed-based reproducibility with `torch.Generator`
   - Device handling (CPU/CUDA)

3. **Plugin Integration** (lines 329-429)
   - `generate_and_save()`: Full pipeline with 4 plugin hooks
   - Hook calling order: start → complete → before_save → after_save
   - Parameter modification by plugins
   - Auto-generated output paths

4. **Configuration** (lines 140-153)
   - Dtype mapping (bfloat16/float16/float32)
   - Model optimization flags (attention slicing, CPU offload, compilation)
   - Device selection and fallback

### QwenImageEditAdapter (738 lines)

**Model Type:** Image-to-Image (Editing)
**Base Class:** ModelAdapterBase
**Key Dependencies:** `diffusers.QwenImageEditPlusPipeline`, `torch`, `PIL`

**Critical Paths to Test:**
1. **Complex Model Loading** (lines 219-437)
   - Standard loading (`from_pretrained`)
   - FP8 hybrid loading (aidiffuser repos)
   - Auto CPU offloading based on VRAM detection
   - Fallback handling when OOM occurs

2. **Image Preprocessing** (lines 173-217)
   - `_preprocess_image()`: Resizing, EXIF orientation, RGB conversion
   - Multi-image handling (1-3 images for composition)
   - Max size constraints

3. **Editing Generation** (lines 492-616)
   - `generate()`: Instruction-based editing
   - Multi-image composition support
   - Parameter validation (input_image, instruction required)
   - Memory management (`_clear_gpu_memory()`)

4. **Plugin Integration** (lines 618-733)
   - `generate_and_save()`: Editing pipeline with plugin hooks
   - Auto-generated filenames with instruction prefix
   - Path sanitization

---

## Part 2: Testing Strategy

### Approach Overview

We'll use a **two-tier testing strategy**:

1. **Unit Tests (Fast, CI-friendly)** → Target: 80-90% coverage
   - Mock all HuggingFace pipelines
   - Mock torch operations
   - Test business logic, parameter handling, error paths
   - Run in CI on every commit

2. **Integration Tests (Slow, Optional)** → Target: Basic smoke tests
   - Use real models (marked with `@pytest.mark.requires_model`)
   - Run manually or on schedule
   - Skip in CI to avoid model downloads

### Why This Works

✅ **Validates adapter logic** without downloading 12-58GB models
✅ **Fast CI** (seconds, not minutes)
✅ **High coverage** of business logic
✅ **Catches bugs** in parameter handling, error handling, plugin integration
✅ **Industry standard** approach used by HuggingFace and other ML projects

---

## Part 3: Implementation Plan

### Phase 1: Create Mock Infrastructure (Week 1)

#### Create `tests/unit/test_model_adapters.py`

**Mock Objects to Create:**

```python
class MockZImagePipeline:
    """Mock for ZImagePipeline that simulates the interface."""

    def __init__(self, *args, **kwargs):
        self.transformer = MagicMock()
        self.device = None
        self._called_with = {}

    def to(self, device):
        self.device = device
        return self

    def enable_model_cpu_offload(self):
        self._called_with['cpu_offload'] = True

    def enable_attention_slicing(self):
        self._called_with['attention_slicing'] = True

    def __call__(self, prompt, height, width, num_inference_steps,
                 guidance_scale, generator=None):
        # Store call parameters for verification
        self._called_with.update({
            'prompt': prompt,
            'height': height,
            'width': width,
            'steps': num_inference_steps,
            'guidance': guidance_scale,
            'generator': generator
        })

        # Return mock output with same structure as real pipeline
        mock_output = MagicMock()
        mock_output.images = [Image.new("RGB", (width, height))]
        return mock_output


class MockQwenImageEditPipeline:
    """Mock for QwenImageEditPlusPipeline."""

    def __init__(self, *args, **kwargs):
        self.transformer = MagicMock()
        self.device = None
        self._called_with = {}

    def to(self, device):
        self.device = device
        return self

    def enable_sequential_cpu_offload(self):
        self._called_with['sequential_offload'] = True

    def enable_model_cpu_offload(self):
        self._called_with['model_offload'] = True

    def enable_attention_slicing(self):
        self._called_with['attention_slicing'] = True

    def enable_xformers_memory_efficient_attention(self):
        self._called_with['xformers'] = True

    def __call__(self, image, prompt, num_inference_steps,
                 guidance_scale, true_cfg_scale, negative_prompt,
                 generator=None, num_images_per_prompt=1):
        self._called_with.update({
            'image': image,
            'prompt': prompt,
            'steps': num_inference_steps,
            'guidance': guidance_scale,
            'true_cfg': true_cfg_scale,
            'neg_prompt': negative_prompt,
            'generator': generator
        })

        # Return mock output
        mock_output = MagicMock()
        output_size = image[0].size if isinstance(image, list) else image.size
        mock_output.images = [Image.new("RGB", output_size)]
        return mock_output
```

### Phase 2: Test ZImageTurboAdapter (Week 1-2)

**Priority 1: Model Lifecycle Tests**

```python
@pytest.mark.unit
class TestZImageTurboAdapterLifecycle:
    """Unit tests for ZImageTurboAdapter model lifecycle."""

    @pytest.fixture
    def config(self, tmp_path):
        return PipeworksConfig(
            device="cpu",
            torch_dtype="float32",
            zimage_model_id="mock/zimage-turbo",
            outputs_dir=tmp_path / "outputs",
            models_dir=tmp_path / "models",
            compile_model=False,
            enable_model_cpu_offload=False,
            enable_attention_slicing=False,
        )

    @pytest.fixture
    def adapter(self, config):
        return ZImageTurboAdapter(config)

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_load_model_success(self, mock_from_pretrained, adapter):
        """Test successful model loading."""
        mock_from_pretrained.return_value = MockZImagePipeline()

        adapter.load_model()

        assert adapter.is_loaded
        assert adapter.pipe is not None
        mock_from_pretrained.assert_called_once()

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_load_model_with_bfloat16(self, mock_from_pretrained, config):
        """Test model loading with bfloat16 dtype."""
        config.torch_dtype = "bfloat16"
        adapter = ZImageTurboAdapter(config)
        mock_from_pretrained.return_value = MockZImagePipeline()

        adapter.load_model()

        call_kwargs = mock_from_pretrained.call_args.kwargs
        assert call_kwargs["torch_dtype"] == torch.bfloat16

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_load_model_already_loaded_skips(self, mock_from_pretrained, adapter):
        """Test that loading an already-loaded model is skipped."""
        mock_from_pretrained.return_value = MockZImagePipeline()

        adapter.load_model()
        adapter.load_model()  # Second call

        # Should only be called once
        mock_from_pretrained.assert_called_once()

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_load_model_failure_not_marked_loaded(self, mock_from_pretrained, adapter):
        """Test that failed loading doesn't mark model as loaded."""
        mock_from_pretrained.side_effect = RuntimeError("Model not found")

        with pytest.raises(RuntimeError):
            adapter.load_model()

        assert not adapter.is_loaded

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_unload_model(self, mock_from_pretrained, adapter):
        """Test model unloading."""
        mock_from_pretrained.return_value = MockZImagePipeline()
        adapter.load_model()

        adapter.unload_model()

        assert not adapter.is_loaded
        assert adapter.pipe is None

    def test_unload_model_when_not_loaded_safe(self, adapter):
        """Test that unloading when not loaded is safe."""
        adapter.unload_model()  # Should not raise
        assert not adapter.is_loaded
```

**Priority 2: Generation Tests**

```python
@pytest.mark.unit
class TestZImageTurboAdapterGeneration:
    """Unit tests for image generation."""

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_generate_returns_pil_image(self, mock_from_pretrained, adapter):
        """Test that generate returns a PIL Image."""
        mock_from_pretrained.return_value = MockZImagePipeline()
        adapter.load_model()

        image = adapter.generate(prompt="test", seed=42)

        assert isinstance(image, Image.Image)
        assert image.size == (1024, 1024)  # Default size from config

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_generate_with_custom_dimensions(self, mock_from_pretrained, adapter):
        """Test generation with custom dimensions."""
        mock_from_pretrained.return_value = MockZImagePipeline()
        adapter.load_model()

        image = adapter.generate(
            prompt="test",
            width=768,
            height=512,
            seed=42
        )

        assert image.size == (768, 512)

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_guidance_scale_forced_to_zero(self, mock_from_pretrained, adapter, caplog):
        """Test that guidance_scale is forced to 0.0 for Turbo."""
        mock_pipeline = MockZImagePipeline()
        mock_from_pretrained.return_value = mock_pipeline
        adapter.load_model()

        adapter.generate(prompt="test", guidance_scale=7.5)

        # Check that warning was logged
        assert "must be 0.0 for Turbo" in caplog.text
        # Verify pipeline was called with 0.0
        assert mock_pipeline._called_with['guidance'] == 0.0

    @patch("diffusers.ZImagePipeline.from_pretrained")
    @patch("torch.Generator")
    def test_generator_created_with_seed(self, mock_generator_class,
                                         mock_from_pretrained, adapter):
        """Test that torch.Generator is created with the provided seed."""
        mock_generator_instance = MagicMock()
        mock_generator_class.return_value = mock_generator_instance
        mock_from_pretrained.return_value = MockZImagePipeline()
        adapter.load_model()

        adapter.generate(prompt="test", seed=42)

        mock_generator_class.assert_called_once_with("cpu")
        mock_generator_instance.manual_seed.assert_called_once_with(42)

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_generate_auto_loads_model(self, mock_from_pretrained, adapter):
        """Test that generate auto-loads model if not loaded."""
        mock_from_pretrained.return_value = MockZImagePipeline()

        # Don't call load_model() explicitly
        image = adapter.generate(prompt="test", seed=42)

        assert adapter.is_loaded
        assert isinstance(image, Image.Image)
```

**Priority 3: Plugin Integration Tests**

```python
@pytest.mark.unit
class TestZImageTurboAdapterPlugins:
    """Unit tests for plugin integration."""

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_generate_and_save_calls_all_plugin_hooks(self,
                                                       mock_from_pretrained,
                                                       adapter, tmp_path):
        """Test that all plugin hooks are called in order."""
        mock_from_pretrained.return_value = MockZImagePipeline()

        # Create mock plugin
        mock_plugin = MagicMock()
        mock_plugin.enabled = True
        mock_plugin.on_generate_start.return_value = {
            "prompt": "test", "seed": 42, "width": 1024, "height": 1024
        }
        mock_plugin.on_generate_complete.side_effect = lambda img, p: img
        mock_plugin.on_before_save.side_effect = lambda img, path, p: (img, path)

        adapter.plugins = [mock_plugin]
        adapter.load_model()

        image, path = adapter.generate_and_save(prompt="test", seed=42)

        # Verify all hooks were called
        mock_plugin.on_generate_start.assert_called_once()
        mock_plugin.on_generate_complete.assert_called_once()
        mock_plugin.on_before_save.assert_called_once()
        mock_plugin.on_after_save.assert_called_once()

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_disabled_plugin_not_called(self, mock_from_pretrained, adapter):
        """Test that disabled plugins are not called."""
        mock_from_pretrained.return_value = MockZImagePipeline()

        mock_plugin = MagicMock()
        mock_plugin.enabled = False
        adapter.plugins = [mock_plugin]
        adapter.load_model()

        adapter.generate_and_save(prompt="test", seed=42)

        mock_plugin.on_generate_start.assert_not_called()

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_plugin_can_modify_parameters(self, mock_from_pretrained, adapter):
        """Test that plugins can modify generation parameters."""
        mock_pipeline = MockZImagePipeline()
        mock_from_pretrained.return_value = mock_pipeline

        # Plugin that modifies prompt
        mock_plugin = MagicMock()
        mock_plugin.enabled = True

        def modify_params(params):
            params = params.copy()
            params['prompt'] = "modified prompt"
            return params

        mock_plugin.on_generate_start.side_effect = modify_params
        adapter.plugins = [mock_plugin]
        adapter.load_model()

        adapter.generate_and_save(prompt="original", seed=42)

        # Check that modified prompt was used
        assert mock_pipeline._called_with['prompt'] == "modified prompt"
```

**Priority 4: Configuration & Optimization Tests**

```python
@pytest.mark.unit
class TestZImageTurboAdapterConfiguration:
    """Unit tests for configuration handling."""

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_cpu_offloading_enabled(self, mock_from_pretrained, config):
        """Test that CPU offloading is enabled when configured."""
        config.enable_model_cpu_offload = True
        mock_pipeline = MockZImagePipeline()
        mock_from_pretrained.return_value = mock_pipeline

        adapter = ZImageTurboAdapter(config)
        adapter.load_model()

        assert mock_pipeline._called_with.get('cpu_offload') is True

    @patch("diffusers.ZImagePipeline.from_pretrained")
    def test_attention_slicing_enabled(self, mock_from_pretrained, config):
        """Test that attention slicing is enabled when configured."""
        config.enable_attention_slicing = True
        mock_pipeline = MockZImagePipeline()
        mock_from_pretrained.return_value = mock_pipeline

        adapter = ZImageTurboAdapter(config)
        adapter.load_model()

        assert mock_pipeline._called_with.get('attention_slicing') is True
```

### Phase 3: Test QwenImageEditAdapter (Week 2-3)

**Priority 1: Preprocessing Tests**

```python
@pytest.mark.unit
class TestQwenImageEditAdapterPreprocessing:
    """Unit tests for image preprocessing."""

    def test_preprocess_image_resizes_large_image(self, adapter):
        """Test that large images are resized."""
        # Create a large test image
        large_img = Image.new("RGB", (2048, 2048))

        processed = adapter._preprocess_image(large_img, max_size=1024)

        assert max(processed.size) <= 1024
        assert processed.mode == "RGB"

    def test_preprocess_image_converts_to_rgb(self, adapter):
        """Test that non-RGB images are converted."""
        rgba_img = Image.new("RGBA", (512, 512))

        processed = adapter._preprocess_image(rgba_img)

        assert processed.mode == "RGB"

    def test_preprocess_image_maintains_aspect_ratio(self, adapter):
        """Test that aspect ratio is maintained during resize."""
        img = Image.new("RGB", (1600, 800))  # 2:1 aspect ratio

        processed = adapter._preprocess_image(img, max_size=1024)

        # Should be 1024x512 (maintaining 2:1 ratio)
        assert processed.size == (1024, 512)
```

**Priority 2: Multi-Image Validation Tests**

```python
@pytest.mark.unit
class TestQwenImageEditAdapterValidation:
    """Unit tests for input validation."""

    @patch("diffusers.QwenImageEditPlusPipeline.from_pretrained")
    def test_generate_requires_input_image(self, mock_from_pretrained, adapter):
        """Test that generate raises error without input image."""
        mock_from_pretrained.return_value = MockQwenImageEditPipeline()
        adapter.load_model()

        with pytest.raises(ValueError, match="input_image is required"):
            adapter.generate(instruction="test")

    @patch("diffusers.QwenImageEditPlusPipeline.from_pretrained")
    def test_generate_requires_instruction(self, mock_from_pretrained, adapter):
        """Test that generate raises error without instruction."""
        mock_from_pretrained.return_value = MockQwenImageEditPipeline()
        adapter.load_model()
        img = Image.new("RGB", (512, 512))

        with pytest.raises(ValueError, match="instruction is required"):
            adapter.generate(input_image=img, instruction="")

    @patch("diffusers.QwenImageEditPlusPipeline.from_pretrained")
    def test_generate_rejects_too_many_images(self, mock_from_pretrained, adapter):
        """Test that generate rejects more than 3 images."""
        mock_from_pretrained.return_value = MockQwenImageEditPipeline()
        adapter.load_model()
        images = [Image.new("RGB", (512, 512)) for _ in range(4)]

        with pytest.raises(ValueError, match="Maximum of 3 input images"):
            adapter.generate(input_image=images, instruction="test")
```

**Priority 3: FP8 Loading Logic Tests**

```python
@pytest.mark.unit
class TestQwenImageEditAdapterFP8Loading:
    """Unit tests for FP8 hybrid loading."""

    @patch("diffusers.QwenImageEditPlusPipeline.from_single_file")
    @patch("safetensors.torch.load_file")
    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.snapshot_download")
    def test_aidiffuser_repo_uses_fp8_loading(self,
                                               mock_snapshot,
                                               mock_hf_download,
                                               mock_load_file,
                                               mock_from_single,
                                               config):
        """Test that aidiffuser repos trigger FP8 loading path."""
        config.qwen_model_id = "aidiffuser/qwen-fp8"
        adapter = QwenImageEditAdapter(config)

        mock_snapshot.return_value = "/fake/config/path"
        mock_hf_download.return_value = "/fake/weights.safetensors"
        mock_load_file.return_value = {"key": torch.tensor([1.0])}
        mock_from_single.return_value = MockQwenImageEditPipeline()

        adapter.load_model()

        # Verify FP8 loading path was taken
        mock_snapshot.assert_called_once()
        mock_hf_download.assert_called_once()
        mock_from_single.assert_called_once()
```

### Phase 4: Integration Tests (Optional, Week 3)

Create `tests/integration/test_model_adapters_integration.py`:

```python
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_model
class TestZImageTurboAdapterIntegration:
    """Integration tests with real models (slow, run manually)."""

    @pytest.mark.skipif(
        os.environ.get("HF_HUB_OFFLINE") == "1",
        reason="Skipping real model test in offline mode"
    )
    def test_real_generation(self):
        """Test actual image generation with real model."""
        config = PipeworksConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="bfloat16",
            zimage_model_id="Tongyi-MAI/Z-Image-Turbo",
        )
        adapter = ZImageTurboAdapter(config)

        try:
            image = adapter.generate(
                prompt="a serene mountain landscape",
                seed=42,
                num_inference_steps=9
            )

            assert isinstance(image, Image.Image)
            assert image.size == (1024, 1024)
        finally:
            adapter.unload_model()
```

---

## Part 4: Test Organization

### Directory Structure

```
tests/
├── unit/
│   ├── test_adapters.py              # Existing UI adapter tests
│   ├── test_model_adapters.py         # NEW: Model adapter unit tests
│   │   ├── Mock classes
│   │   ├── TestZImageTurboAdapterLifecycle
│   │   ├── TestZImageTurboAdapterGeneration
│   │   ├── TestZImageTurboAdapterPlugins
│   │   ├── TestZImageTurboAdapterConfiguration
│   │   ├── TestQwenImageEditAdapterPreprocessing
│   │   ├── TestQwenImageEditAdapterValidation
│   │   ├── TestQwenImageEditAdapterFP8Loading
│   │   └── TestQwenImageEditAdapterConfiguration
│   └── ...
├── integration/
│   ├── test_model_adapters_integration.py  # NEW: Real model tests
│   │   ├── TestZImageTurboAdapterIntegration
│   │   └── TestQwenImageEditAdapterIntegration
│   └── ...
├── fixtures/
│   └── test_images/                   # Sample images for testing
│       ├── test_512x512.png
│       ├── test_1024x1024.png
│       └── test_rgba.png
└── conftest.py
```

### Pytest Configuration

Update `pytest.ini`:

```ini
[pytest]
markers =
    unit: Fast unit tests with mocks (default)
    integration: Integration tests (may require external resources)
    slow: Slow tests (several seconds+)
    requires_model: Tests that download real ML models (skip in CI)

# Run only unit tests by default
addopts = -m "not slow and not requires_model"
```

### CI Configuration

Update `.github/workflows/ci.yml`:

```yaml
- name: Run unit tests (fast, mocked)
  run: |
    pytest tests/unit/ -m "not slow" --cov=src/pipeworks/core/adapters --cov-report=xml
  env:
    HF_HUB_OFFLINE: 1  # Prevent accidental model downloads

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    flags: adapters
```

---

## Part 5: Coverage Goals by Component

### ZImageTurboAdapter

| Component | Lines | Current | Target | Strategy |
|-----------|-------|---------|--------|----------|
| `__init__` | 13 | 0% | 100% | Simple instantiation test |
| `load_model` | 63 | 0% | 90% | Mock `from_pretrained`, test all paths |
| `generate` | 54 | 0% | 95% | Mock pipeline, test parameter handling |
| `generate_and_save` | 68 | 0% | 90% | Mock pipeline + plugins, test hook order |
| `unload_model` | 12 | 0% | 100% | Simple state management test |
| `is_loaded` | 4 | 0% | 100% | Property test |
| **Total** | **214** | **18.75%** | **90%** | **~35 test cases** |

### QwenImageEditAdapter

| Component | Lines | Current | Target | Strategy |
|-----------|-------|---------|--------|----------|
| `__init__` | 17 | 0% | 100% | Simple instantiation test |
| `load_model` | 193 | 0% | 75% | Mock pipelines, test FP8 path, standard path |
| `_preprocess_image` | 25 | 0% | 95% | Test resize, RGB conversion, aspect ratio |
| `_clear_gpu_memory` | 4 | 0% | 100% | Simple mock test |
| `generate` | 91 | 0% | 85% | Mock pipeline, test validation, multi-image |
| `generate_and_save` | 75 | 0% | 85% | Mock pipeline + plugins, test paths |
| `unload_model` | 22 | 0% | 95% | Test cleanup, error handling |
| `is_loaded` | 3 | 0% | 100% | Property test |
| **Total** | **430** | **10.08%** | **80%** | **~40 test cases** |

---

## Part 6: Implementation Timeline

### Week 1: Foundation
- ✅ Create mock infrastructure (MockZImagePipeline, MockQwenImageEditPipeline)
- ✅ Set up test file structure
- ✅ Write 10 tests for ZImageTurboAdapter lifecycle
- ✅ Write 5 tests for ZImageTurboAdapter generation
- **Target: 30% coverage on ZImageTurboAdapter**

### Week 2: Core Coverage
- ✅ Write 10 tests for ZImageTurboAdapter plugins
- ✅ Write 5 tests for ZImageTurboAdapter configuration
- ✅ Write 10 tests for QwenImageEditAdapter preprocessing/validation
- ✅ Write 8 tests for QwenImageEditAdapter generation
- **Target: 80% coverage on ZImageTurboAdapter, 40% coverage on QwenImageEditAdapter**

### Week 3: Complete & Polish
- ✅ Write 10 tests for QwenImageEditAdapter FP8 loading
- ✅ Write 8 tests for QwenImageEditAdapter plugins
- ✅ Add integration test skeleton (marked to skip in CI)
- ✅ Review and improve test coverage gaps
- ✅ Update CI configuration
- **Target: 90% coverage on ZImageTurboAdapter, 80% coverage on QwenImageEditAdapter**

### Week 4: Documentation & Maintenance
- ✅ Document testing patterns in TESTING.md
- ✅ Add coverage badges to README
- ✅ Create PR with full test suite
- ✅ Set up scheduled integration test runs (optional)

---

## Part 7: Success Metrics

### Quantitative Goals

- ✅ **ZImageTurboAdapter**: 18.75% → 90% coverage (+71 percentage points)
- ✅ **QwenImageEditAdapter**: 10.08% → 80% coverage (+70 percentage points)
- ✅ **CI Time**: Keep adapter tests under 5 minutes
- ✅ **Test Count**: ~75 unit tests total
- ✅ **Reliability**: 100% pass rate in CI (no flaky tests)

### Qualitative Goals

- ✅ **Confidence**: Can refactor adapters without fear of breaking functionality
- ✅ **Documentation**: Tests serve as usage examples
- ✅ **Maintainability**: Easy to add tests for new adapters
- ✅ **Developer Experience**: Fast feedback loop during development

---

## Part 8: Common Pitfalls to Avoid

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Over-mocking** | Tests don't catch real issues | Only mock external dependencies (diffusers, torch.cuda), not your own code |
| **Under-mocking** | CI downloads models | Always mock `from_pretrained` and `from_single_file` |
| **Brittle tests** | Tests break on minor changes | Test behavior, not implementation details |
| **Missing error paths** | Bugs in error handling go undetected | Use `side_effect` to test exceptions |
| **Slow tests** | CI takes too long | Use `pytest.mark.slow` for any test >1 second |
| **Incomplete mocks** | Tests pass but real code fails | Make mocks match real interface (return types, call signatures) |

---

## Part 9: Next Steps

1. **Review this plan** - Ensure alignment with team
2. **Create test file skeleton** - Set up basic structure
3. **Write first 5 tests** - Validate mocking approach
4. **Run coverage check** - Verify coverage is improving
5. **Iterate** - Add more tests following the patterns
6. **Monitor CI** - Ensure tests remain fast and reliable
7. **Update documentation** - Document testing patterns for future adapters

---

## Conclusion

This plan provides a **realistic, battle-tested approach** to improving test coverage for model adapters without sacrificing CI speed or reliability. By using mocks strategically, we can achieve 80-90% coverage while keeping tests fast and maintainable.

The key insight is that **we don't need to test the HuggingFace models**—they're already tested by HuggingFace. What we need to test is **our adapter code**: parameter handling, error handling, plugin integration, and business logic.

**Estimated Effort:** 3-4 weeks for one developer
**Expected Outcome:** 80-90% coverage, <5 minute CI time, robust test suite

---

## References

- [Testing Strategy for ML Model Adapters Guide](Testing Strategy for ML Model Adapters_ A Comprehensive Guide.md)
- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Codecov best practices](https://docs.codecov.com/docs)
