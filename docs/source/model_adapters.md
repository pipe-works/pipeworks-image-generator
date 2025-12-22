# Model Adapter System

## Overview

Pipeworks now supports multiple AI models through a flexible **Model Adapter** architecture. This allows you to use different models for different tasks:

- **Z-Image-Turbo**: Fast text-to-image generation for asset creation
- **Qwen-Image-Edit**: Instruction-based image editing and refinement
- **More models**: Easy to add Flux, SDXL, ControlNet, etc.

## Architecture

The model adapter system consists of:

1. **ModelAdapterBase**: Abstract base class defining the interface all models must implement
2. **Model Registry**: Central registry for discovering and instantiating model adapters
3. **Specific Adapters**: Model-specific implementations (ZImageTurboAdapter, QwenImageEditAdapter, etc.)
4. **Workflows**: Can specify which model adapter they require
5. **UI Integration**: Dynamic model switching in the Gradio interface

```
┌─────────────────────────────────────────┐
│      Model Registry                     │
│  - Manages available models             │
│  - Instantiates adapters                │
└─────────────┬───────────────────────────┘
              │
              ├──> ZImageTurboAdapter (text-to-image)
              │     └─> Z-Image-Turbo model
              │
              ├──> QwenImageEditAdapter (image-edit)
              │     └─> Qwen-Image-Edit model
              │
              └──> [Future adapters...]
                    └─> Flux, SDXL, ControlNet, etc.
```

## Basic Usage

### Programmatic API

#### Text-to-Image Generation (Z-Image-Turbo)

```python
from pipeworks.core import model_registry, config

# Instantiate the model adapter
adapter = model_registry.instantiate("Z-Image-Turbo", config)

# Generate an image
image = adapter.generate(
    prompt="a fantasy sword sprite, pixel art style",
    width=1024,
    height=1024,
    seed=42
)

# Save the image
image.save("sword.png")

# Or generate and save in one step
image, path = adapter.generate_and_save(
    prompt="a fantasy sword sprite, pixel art style",
    seed=42
)
print(f"Saved to: {path}")
```

#### Image Editing (Qwen-Image-Edit)

```python
from pipeworks.core import model_registry, config
from PIL import Image

# Load base image
base_image = Image.open("character.png")

# Instantiate the editing model
editor = model_registry.instantiate("Qwen-Image-Edit", config)

# Edit the image with natural language
edited = editor.generate(
    input_image=base_image,
    instruction="change the character's hair color to blue",
    seed=42
)

# Save edited image
edited, path = editor.generate_and_save(
    input_image=base_image,
    instruction="add a glowing sword to the character's hand",
    seed=42
)
print(f"Edited image saved to: {path}")
```

### Using with Workflows

Workflows can specify which model adapter they require:

```python
from pipeworks.workflows.base import WorkflowBase, workflow_registry
from pipeworks.core import model_registry, config

class AssetGenerationWorkflow(WorkflowBase):
    name = "Asset Generation"
    description = "Generate game assets"
    model_adapter_name = "Z-Image-Turbo"  # Specify required model
    model_type = "text-to-image"

    def build_prompt(self, asset_type: str, style: str, **kwargs) -> str:
        return f"{asset_type}, {style}, game asset, clean background"

# Register the workflow
workflow_registry.register(AssetGenerationWorkflow)

# Use the workflow
workflow = workflow_registry.instantiate("Asset Generation")

# Attach the appropriate model adapter
adapter = model_registry.instantiate(workflow.model_adapter_name, config)
workflow.set_model_adapter(adapter)

# Generate using workflow
image, params = workflow.generate(asset_type="sword", style="pixel art")
```

### Plugin System Integration

Model adapters work seamlessly with the plugin system:

```python
from pipeworks.core import model_registry, config
from pipeworks.plugins.base import plugin_registry

# Create plugins
metadata_plugin = plugin_registry.instantiate("Save Metadata")

# Pass plugins to model adapter
adapter = model_registry.instantiate(
    "Z-Image-Turbo",
    config,
    plugins=[metadata_plugin]
)

# Plugins are automatically called during generation
image, path = adapter.generate_and_save(
    prompt="test",
    seed=42
)
# Metadata JSON file is automatically created by plugin
```

## Configuration

### Environment Variables

Add model-specific configuration to your `.env` file:

```bash
# Default model adapter to use
PIPEWORKS_DEFAULT_MODEL_ADAPTER=Z-Image-Turbo

# Model-specific HuggingFace IDs
PIPEWORKS_ZIMAGE_MODEL_ID=Tongyi-MAI/Z-Image-Turbo
PIPEWORKS_QWEN_MODEL_ID=Qwen/Qwen-Image-Edit-2509

# General model settings (shared across adapters)
PIPEWORKS_TORCH_DTYPE=bfloat16
PIPEWORKS_DEVICE=cuda
PIPEWORKS_NUM_INFERENCE_STEPS=9

# Performance optimizations
PIPEWORKS_ENABLE_ATTENTION_SLICING=false
PIPEWORKS_COMPILE_MODEL=false
```

### Python Configuration

```python
from pipeworks.core.config import PipeworksConfig

config = PipeworksConfig(
    default_model_adapter="Qwen-Image-Edit",
    qwen_model_id="Qwen/Qwen-Image-Edit-2509",
    device="cuda",
    torch_dtype="bfloat16"
)
```

## UI Integration

The Gradio UI has been updated to support dynamic model switching:

### UI State Management

```python
from pipeworks.ui.state import initialize_ui_state, switch_model

# Initialize with default model
state = initialize_ui_state()
print(f"Current model: {state.current_model_name}")

# Switch to a different model
state = switch_model(state, "Qwen-Image-Edit")
print(f"Switched to: {state.current_model_name}")

# Plugins are preserved when switching models
```

### Model Selection Dropdown (Future Enhancement)

The UI can be enhanced to include a model selection dropdown that allows users to switch between models on the fly. This would call `switch_model()` to dynamically load the selected model.

## Interactive Fiction Game Use Case

For your interactive fiction game, you can use different models for different purposes:

```python
from pipeworks.core import model_registry, config

# Asset generation with Z-Image-Turbo
turbo = model_registry.instantiate("Z-Image-Turbo", config)
character_sprite = turbo.generate(
    prompt="fantasy character sprite, pixel art, front view",
    width=512,
    height=512,
    seed=42
)

# Image refinement with Qwen-Image-Edit
qwen = model_registry.instantiate("Qwen-Image-Edit", config)
refined_sprite = qwen.generate(
    input_image=character_sprite,
    instruction="add a red cloak and staff",
    seed=43
)

# Generate background
background = turbo.generate(
    prompt="fantasy forest background, game art, parallax layer",
    width=1920,
    height=1080,
    seed=44
)
```

## Adding New Model Adapters

To add support for a new model:

1. **Create the adapter class** in `src/pipeworks/core/adapters/`:

```python
from pipeworks.core.model_adapters import ModelAdapterBase, model_registry

class FluxAdapter(ModelAdapterBase):
    name = "Flux"
    description = "High-quality text-to-image with Flux"
    model_type = "text-to-image"
    version = "1.0.0"

    def __init__(self, config, plugins=None):
        super().__init__(config, plugins)
        self.model_id = getattr(config, "flux_model_id", "black-forest-labs/FLUX.1-dev")
        self.pipe = None
        self._model_loaded = False

    def load_model(self):
        # Implement model loading
        pass

    def generate(self, prompt, **kwargs):
        # Implement generation
        pass

    def generate_and_save(self, **kwargs):
        # Implement save with plugin hooks
        pass

    def unload_model(self):
        # Implement cleanup
        pass

    @property
    def is_loaded(self):
        return self._model_loaded

# Register the adapter
model_registry.register(FluxAdapter)
```

2. **Import in** `src/pipeworks/core/adapters/__init__.py`:

```python
from .flux import FluxAdapter

__all__ = ["ZImageTurboAdapter", "QwenImageEditAdapter", "FluxAdapter"]
```

3. **Add configuration** in `src/pipeworks/core/config.py`:

```python
flux_model_id: str = Field(
    default="black-forest-labs/FLUX.1-dev",
    description="HuggingFace model ID for Flux"
)
```

4. **Update** `.env.example`:

```bash
PIPEWORKS_FLUX_MODEL_ID=black-forest-labs/FLUX.1-dev
```

## Model Adapter Interface

All model adapters must implement these methods:

### Required Methods

```python
def load_model(self) -> None:
    """Load the model into memory."""
    pass

def generate(self, **kwargs) -> Image.Image:
    """Generate or edit an image."""
    pass

def generate_and_save(self, **kwargs) -> tuple[Image.Image, Path]:
    """Generate and save with plugin hooks."""
    pass

def unload_model(self) -> None:
    """Unload model from memory."""
    pass

@property
def is_loaded(self) -> bool:
    """Check if model is loaded."""
    pass
```

### Model Types

Specify the type of model in your adapter:

- `"text-to-image"`: Generate images from text prompts
- `"image-edit"`: Edit existing images with instructions
- `"img2img"`: Transform images with prompts
- `"inpainting"`: Fill masked regions of images

### Plugin Lifecycle Hooks

Model adapters should call these plugin hooks in `generate_and_save()`:

1. **on_generate_start(params)**: Before generation (can modify params)
2. **on_generate_complete(image, params)**: After generation (can modify image)
3. **on_before_save(image, path, params)**: Before saving (can modify image/path)
4. **on_after_save(image, path, params)**: After saving (e.g., metadata export)

## Migration from Legacy ImageGenerator

### Old Code (Legacy)

```python
from pipeworks.core import ImageGenerator, config

generator = ImageGenerator(config)
image, path = generator.generate_and_save(
    prompt="test",
    seed=42
)
```

### New Code (Model Adapters)

```python
from pipeworks.core import model_registry, config

adapter = model_registry.instantiate("Z-Image-Turbo", config)
image, path = adapter.generate_and_save(
    prompt="test",
    seed=42
)
```

### Backward Compatibility

The `ImageGenerator` class is still available for backward compatibility, but internally it now uses the Z-Image-Turbo adapter. New code should use the model adapter system directly.

```python
# This still works but is deprecated
from pipeworks.core import ImageGenerator
generator = ImageGenerator(config)

# Prefer this instead
from pipeworks.core import model_registry
adapter = model_registry.instantiate("Z-Image-Turbo", config)
```

## Available Models

### Z-Image-Turbo
- **Type**: text-to-image
- **Best for**: Fast asset generation, sprites, icons, backgrounds
- **Speed**: Very fast (9 steps optimal)
- **Quality**: Good for game assets
- **Model ID**: `Tongyi-MAI/Z-Image-Turbo`
- **Requirements**:
  - guidance_scale must be 0.0 (automatically enforced)
  - Optimal steps: 9

### Qwen-Image-Edit
- **Type**: image-edit
- **Best for**: Refining existing images, adding details, style transfer
- **Speed**: Moderate (20-50 steps)
- **Quality**: High-quality edits
- **Model ID**: `Qwen/Qwen-Image-Edit-2509`
- **Requirements**:
  - Requires input image
  - Natural language instructions

## Troubleshooting

### Model not loading

```python
# Check available models
from pipeworks.core import model_registry
print(model_registry.list_available())

# Get model info
info = model_registry.get_adapter_info("Z-Image-Turbo")
print(info)

# Check if model is loaded
adapter = model_registry.instantiate("Z-Image-Turbo", config)
print(f"Loaded: {adapter.is_loaded}")
```

### Model type mismatch

If you see a warning about model type mismatch:

```
Model adapter type mismatch: workflow 'Asset Generation' expects
'text-to-image' but got 'image-edit'
```

This means you're using an image editing model for a workflow that expects text-to-image. Use the correct model adapter:

```python
# Wrong - AssetWorkflow expects text-to-image
editor = model_registry.instantiate("Qwen-Image-Edit", config)
workflow.set_model_adapter(editor)  # Warning!

# Correct
generator = model_registry.instantiate("Z-Image-Turbo", config)
workflow.set_model_adapter(generator)  # OK
```

### CUDA out of memory

If you get CUDA OOM errors with multiple models:

```python
# Unload previous model before loading new one
current_adapter.unload_model()

# Or use the switch_model utility (UI)
from pipeworks.ui.state import switch_model
state = switch_model(state, "Qwen-Image-Edit")  # Automatically unloads old model
```

## Performance Tips

1. **Pre-load models**: Load models during initialization to reduce first-generation latency
2. **Reuse adapters**: Keep adapter instances alive and reuse them instead of creating new ones
3. **Unload when switching**: Always unload models before loading a new one to free VRAM
4. **Use appropriate dtype**: bfloat16 is recommended for best quality/performance balance
5. **Enable optimizations**: Consider enabling attention slicing or model compilation for faster inference

## Future Enhancements

Planned improvements to the model adapter system:

- [ ] **UI Model Selector**: Dropdown in Gradio UI for dynamic model switching
- [ ] **Model Profiles**: Predefined configs for different use cases
- [ ] **Automatic Model Selection**: Workflows automatically select best model
- [ ] **Multi-Model Pipelines**: Chain multiple models in sequence
- [ ] **Model Caching**: Keep multiple models loaded with LRU eviction
- [ ] **ControlNet Support**: Add ControlNet adapter for precise control
- [ ] **Img2Img Workflows**: Add img2img adapter for style transfer
- [ ] **Batch Processing**: Process multiple images across different models

## Examples

See the `examples/` directory for complete examples:

- `examples/text_to_image.py`: Basic text-to-image generation
- `examples/image_editing.py`: Image editing workflow
- `examples/multi_model_pipeline.py`: Using multiple models in sequence
- `examples/interactive_fiction.py`: Game asset generation pipeline
