# Pipeworks Image Generator

[![Test and Lint](https://github.com/aa-parky/pipeworks-image-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/aa-parky/pipeworks-image-generator/actions/workflows/ci.yml)
[![codecov](https://app.codecov.io/gh/aa-parky/pipeworks-image-generator/branch/main/graph/badge.svg)](https://app.codecov.io/gh/aa-parky/pipeworks-image-generator)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Programmatic image generation with Z-Image-Turbo and an extensible plugin system.

## Overview

Pipeworks is a Python-based image generation framework designed for programmatic control and experimentation. Moving beyond node-based interfaces like ComfyUI, Pipeworks provides a clean, code-first approach to image generation with a focus on extensibility and automation.

### Key Features

- **Z-Image-Turbo Integration**: Sub-second inference with the state-of-the-art 6B parameter model
- **Gradio UI**: Clean, responsive web interface with generation, gallery browser, and favorites management
- **Programmatic API**: Full Python API for scripting and automation
- **Extensible Architecture**: Plugin system and workflow orchestration for custom generation strategies
- **Advanced Prompt Builder**: File-based prompt construction with random selection and segment composition
- **Gallery Browser**: Browse, filter, favorite, and manage generated images
- **Tokenizer Analysis**: Real-time token count and analysis for prompt optimization
- **Local-First**: Everything runs on your hardware, no cloud dependencies
- **Production-Ready**: Built with modern Python best practices, comprehensive test coverage (50%+), CI/CD pipeline

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 50GB+ free disk space (for model cache)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pipeworks-image-generator.git
cd pipeworks-image-generator
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
# Or install from requirements.txt:
pip install -r requirements.txt
```

4. (Optional) Configure environment:
```bash
cp .env.example .env
# Edit .env with your preferences
```

### Running the Application

Launch the Gradio UI:
```bash
pipeworks
# Or directly:
python -m pipeworks.ui.app
```

The UI will be available at `http://0.0.0.0:7860` (accessible on your local network).

## Usage

### Gradio UI

The web interface provides two main tabs:

#### Generate Tab
1. **Basic Generation**:
   - Open the web interface (default: `http://localhost:7860`)
   - Enter your prompt in the text box
   - Adjust generation parameters (width, height, steps, seed)
   - Click "Generate Image"
   - Generated images are saved to `outputs/` directory

2. **Prompt Builder** (Advanced):
   - Navigate through the `inputs/` directory
   - Select text files containing prompt fragments
   - Choose selection mode (random line, specific line, range, all lines, N random)
   - Compose prompts from three segments (start, middle, end)
   - Build and preview the combined prompt
   - Use tokenizer analysis to stay within token limits

3. **Tokenizer Analysis**:
   - Analyze any prompt for token count
   - View token breakdown and vocabulary usage
   - Ensure prompts stay within the 256 token limit

4. **Plugins**:
   - Enable/disable the "Save Metadata" plugin
   - Metadata exports generation parameters to JSON
   - Choose between compact or pretty format

5. **Workflows** (Coming Soon):
   - Select content-specific workflows (Character, Game Asset, City Map)
   - Workflows provide optimized prompts for specific use cases

#### Gallery Browser Tab
1. **Browse Generated Images**:
   - View all images in `outputs/` or `catalog/` directories
   - Navigate folder structure
   - Click images to view details

2. **Favorites Management**:
   - Click the heart icon to favorite images
   - Filter gallery to show only favorites
   - Favorites are stored in a SQLite database

3. **Catalog Management**:
   - Move favorited images to the `catalog/` directory for archiving
   - Preserves folder structure and metadata

### Programmatic Usage

#### Basic Text-to-Image Generation
```python
from pipeworks import model_registry, config

# Instantiate Z-Image-Turbo adapter
adapter = model_registry.instantiate("Z-Image-Turbo", config)

# Generate image
image = adapter.generate(
    prompt="A serene mountain landscape at sunset",
    width=1024,
    height=1024,
    num_inference_steps=9,
    seed=42,
)

# Generate and save
image, path = adapter.generate_and_save(
    prompt="A cute cat sleeping on a cozy blanket",
    seed=12345,
)
print(f"Saved to: {path}")
```

#### Image Editing with Qwen
```python
from pipeworks import model_registry, config
from PIL import Image

# Instantiate Qwen-Image-Edit adapter
editor = model_registry.instantiate("Qwen-Image-Edit", config)

# Load base image
base_image = Image.open("character.png")

# Edit and save
edited, path = editor.generate_and_save(
    input_image=base_image,
    instruction="change the character's hair color to blue",
    num_inference_steps=40,
    seed=42
)
print(f"Saved to: {path}")
```

#### Using Plugins
```python
from pipeworks import model_registry, config
from pipeworks.plugins.base import plugin_registry

# Instantiate a plugin
metadata_plugin = plugin_registry.instantiate(
    "Save Metadata",
    output_folder="metadata",
    pretty_format=True
)

# Create adapter with plugin
adapter = model_registry.instantiate("Z-Image-Turbo", config, plugins=[metadata_plugin])

# Generate - plugin hooks will run automatically
image, path = adapter.generate_and_save(
    prompt="A beautiful landscape",
    seed=42
)
# Metadata JSON will be saved alongside the image
```

#### Using Prompt Builder
```python
from pipeworks.core.prompt_builder import PromptBuilder
from pipeworks.ui.models import SegmentConfig

# Initialize prompt builder
builder = PromptBuilder(inputs_dir="inputs")

# Browse available files
files = builder.get_file_choices()

# Configure a segment
segment = SegmentConfig(
    file_path="inputs/styles/realistic.txt",
    mode="random_line"
)

# Build prompt from segments
prompt = builder.build_from_segments(
    start=segment,
    middle=None,
    end=None
)
print(f"Generated prompt: {prompt}")
```

#### Using Workflows
```python
from pipeworks import model_registry, config
from pipeworks.workflows.base import workflow_registry

# List available workflows
workflows = workflow_registry.list_available()
print(workflows)  # ['Character', 'Game Asset', 'City Map']

# Instantiate a workflow
character_workflow = workflow_registry.instantiate("Character")

# Build a workflow-specific prompt
prompt = character_workflow.build_prompt(
    character_type="warrior",
    style="fantasy"
)

# Use with adapter
adapter = model_registry.instantiate("Z-Image-Turbo", config)
image = adapter.generate(prompt=prompt, seed=42)
```

## Architecture

```
pipeworks-image-generator/
├── src/pipeworks/
│   ├── core/                    # Core generation engine
│   │   ├── config.py            # Pydantic configuration (env-based)
│   │   ├── model_adapters.py   # Multi-model adapter system + registry
│   │   ├── adapters/            # Model-specific implementations
│   │   │   ├── zimage_turbo.py  # Z-Image-Turbo text-to-image adapter
│   │   │   └── qwen_image_edit.py # Qwen-Image-Edit editing adapter
│   │   ├── prompt_builder.py   # File-based prompt construction
│   │   ├── tokenizer.py         # Tokenization analysis utilities
│   │   ├── character_conditions.py # Procedural character generation
│   │   ├── facial_conditions.py    # Facial signal generation
│   │   ├── gallery_browser.py  # Gallery browsing and filtering
│   │   ├── favorites_db.py     # SQLite favorites database
│   │   └── catalog_manager.py  # Archive management
│   ├── plugins/                 # Plugin system
│   │   ├── base.py              # PluginBase + PluginRegistry
│   │   └── save_metadata.py    # Built-in metadata export plugin
│   ├── workflows/               # Workflow system
│   │   ├── base.py              # WorkflowBase + WorkflowRegistry
│   │   ├── character.py         # Character generation workflow
│   │   ├── game_asset.py        # Game asset workflow
│   │   └── city_map.py          # City/map generation workflow
│   └── ui/                      # Gradio web interface
│       ├── app.py               # Main UI layout and event wiring
│       ├── components.py        # Reusable Gradio component builders
│       ├── handlers/            # Event handler functions
│       ├── models.py            # Pydantic models for UI state
│       ├── state.py             # UI state management
│       ├── validation.py        # Input validation logic
│       ├── formatting.py        # Output formatting utilities
│       └── adapters.py          # UI value conversion functions
├── models/                      # Model cache (gitignored)
├── outputs/                     # Generated images (gitignored)
├── inputs/                      # Prompt builder text files
├── catalog/                     # Archived/favorited images (gitignored)
└── pyproject.toml               # Project configuration
```

### Key Architectural Patterns

1. **Multi-Model Adapter System**: Unified interface for different AI models (text-to-image, image-editing) via `ModelAdapterBase` and `model_registry`
2. **Configuration System**: Pydantic Settings with environment variable loading (`PIPEWORKS_*` prefix)
3. **Plugin Architecture**: Extensible hooks at four lifecycle points (start, complete, before_save, after_save)
4. **Workflow System**: Encapsulates generation strategies for specific content types
5. **UI State Management**: Session-based state using Gradio's `gr.State` with Pydantic models
6. **Prompt Builder**: File-based prompt construction with multiple selection modes and caching
7. **Separation of Concerns**: UI layout (app.py), business logic (handlers/), data models (models.py), validation (validation.py)

## How the Model Works: A Beginner's Guide

Understanding how Pipeworks loads and uses the Z-Image-Turbo model can help you optimize performance and troubleshoot issues. This section explains the model's component parts and how they fit together.

### The Big Picture: From Text to Image

When you generate an image, several AI components work together in a pipeline:

```
Your Prompt → Tokenizer → Text Encoder → Diffusion Transformer → VAE Decoder → Final Image
```

Let's break down each component:

### 1. **Tokenizer** (Text → Numbers)
**What it does**: Converts your text prompt into numbers that the AI can understand.

**Example**:
- Input: `"a cute cat sleeping"`
- Output: `[320, 8472, 2368, 11029]` (token IDs)

**Why it matters**:
- Models have token limits (Z-Image-Turbo: 256 tokens)
- Pipeworks includes a tokenizer analyzer to help you stay within limits
- Each word/punctuation becomes one or more tokens

**In the code** (`core/tokenizer.py`):
```python
# The tokenizer breaks your prompt into pieces the model understands
tokens = tokenizer.encode("a cute cat")
# Result: List of integer IDs representing each word/subword
```

### 2. **Text Encoder** (Numbers → Understanding)
**What it does**: Transforms token IDs into "embeddings" - mathematical representations that capture the *meaning* of your prompt.

**Think of it like**: A dictionary that knows "cat" and "feline" are related concepts, or that "sleeping" implies a peaceful pose.

**Why it matters**:
- This is where the model "understands" what you're asking for
- The quality of these embeddings directly affects image quality
- Z-Image-Turbo uses advanced transformer-based encoding

**Technical detail**: The encoder produces a 768-dimensional vector for each token. These vectors encode semantic relationships learned from billions of image-caption pairs.

### 3. **Diffusion Transformer (DiT)** (Understanding → Noisy Image → Clear Image)
**What it does**: The "brain" of the model. Iteratively refines random noise into a coherent image guided by your prompt.

**The process**:
1. Starts with pure random noise (static)
2. At each step, predicts "what this should look like" based on your prompt
3. Gradually removes noise over 9 steps (for Turbo models)
4. Each step moves closer to the final image

**Why it's called "Turbo"**:
- Traditional diffusion models need 50-100 steps
- Z-Image-Turbo uses distillation to achieve high quality in just 9 steps
- This is why generation is so fast (sub-second vs 30+ seconds)

**Key parameter - `num_inference_steps`**:
- Default: 9 steps (recommended for Turbo)
- More steps ≠ better quality for Turbo models
- Turbo models are optimized for this specific step count

**In the code** (`core/adapters/zimage_turbo.py`):
```python
output = self.pipe(
    prompt=prompt,
    num_inference_steps=9,  # Optimal for Z-Image-Turbo
    guidance_scale=0.0,     # MUST be 0.0 for Turbo
    generator=generator,    # For reproducible results
)
```

### 4. **VAE Decoder** (Latent Space → Pixels)
**What it does**: Converts the compressed "latent" representation into actual pixels you can see.

**Think of it like**:
- The DiT works with a compressed 64x64 representation
- The VAE decoder upscales this to your target resolution (e.g., 1024x1024)
- Like decompressing a ZIP file into the full content

**Why it matters**:
- This step determines final image sharpness and detail
- The VAE is a neural network trained to reconstruct high-quality images
- All diffusion models use this two-stage approach (latent space → pixel space) for efficiency

### Model Loading Process

When you run Pipeworks, here's what happens behind the scenes:

#### Stage 1: Initialization (Instant)
```python
from pipeworks import model_registry, config
adapter = model_registry.instantiate("Z-Image-Turbo", config)
```
- Creates the adapter object
- Loads configuration from environment variables
- **Model is NOT loaded yet** (lazy loading for efficiency)

#### Stage 2: Model Download & Loading (First run: 5-10 minutes, Subsequent: 10-30 seconds)
```python
adapter.load_model()  # Called automatically on first generate()
```

**What gets loaded** (`core/adapters/zimage_turbo.py`):
1. **Pipeline from HuggingFace Hub** (~12GB download)
   - Tokenizer weights
   - Text encoder weights
   - Diffusion transformer weights (6 billion parameters!)
   - VAE decoder weights

2. **Cache location**: `./models/` directory
   - Subsequent runs load from cache (much faster)
   - No re-download needed

3. **Memory precision** (bfloat16 recommended):
   - `bfloat16`: 12GB VRAM, best quality/speed balance
   - `float16`: 12GB VRAM, slightly faster but less stable
   - `float32`: 24GB VRAM, highest precision, slower

#### Stage 3: Device Transfer & Optimization (10-30 seconds)
```python
self.pipe.to(self.config.device)  # Move to GPU
```

**Optimizations applied**:
- **Attention Slicing** (optional): Reduces VRAM usage by processing attention in chunks
- **CPU Offloading** (optional): Moves unused layers to RAM, saves VRAM
- **Model Compilation** (optional): torch.compile speeds up inference (adds 1-2 min first run)
- **Flash Attention** (optional): Faster attention mechanism if supported

#### Stage 4: Generation (Sub-second to 3 seconds)
```python
image = generator.generate(prompt="a cat", seed=42)
```

**Per-generation process**:
1. Tokenize prompt → token IDs
2. Encode tokens → semantic embeddings
3. Create random noise tensor (seeded for reproducibility)
4. Run 9 diffusion steps (DiT iteratively refines noise)
5. Decode latent → final pixel image
6. Return PIL Image object

### Critical Z-Image-Turbo Requirements

**These are hardcoded constraints of the Turbo architecture:**

1. **`guidance_scale` MUST be 0.0**
   - Regular models use guidance_scale (1.0-20.0) to strengthen prompt adherence
   - Turbo models are distilled to work WITHOUT guidance
   - Pipeworks automatically enforces this (`core/adapters/zimage_turbo.py`)

2. **Optimal steps: 9**
   - Results in 8 DiT forward passes (optimal for distillation)
   - More steps won't improve quality (and may degrade it)

3. **Recommended dtype: bfloat16**
   - Float16 can cause numerical instability
   - Float32 requires 2x VRAM with minimal quality gain

### Memory Requirements

**GPU VRAM breakdown** (for 1024x1024 generation):
- Text Encoder: ~2GB
- Diffusion Transformer (6B params): ~8GB
- VAE Decoder: ~1GB
- Working memory (activations): ~2-3GB
- **Total: 13-14GB VRAM** (16GB recommended for headroom)

**CPU RAM**:
- Model loading peaks at ~20GB during initialization
- Steady state: ~8GB

### Performance Tuning

**For low VRAM (12-16GB)**:
```bash
PIPEWORKS_ENABLE_ATTENTION_SLICING=true
PIPEWORKS_TORCH_DTYPE=bfloat16
```

**For maximum speed (16GB+ VRAM)**:
```bash
PIPEWORKS_COMPILE_MODEL=true
PIPEWORKS_ATTENTION_BACKEND=flash
PIPEWORKS_TORCH_DTYPE=bfloat16
```

**For CPU-only (slow, testing only)**:
```bash
PIPEWORKS_DEVICE=cpu
PIPEWORKS_TORCH_DTYPE=float32
```

### Reproducibility: The Role of Seeds

**What is a seed?**
- A seed is a starting number for the random number generator
- Same seed + same prompt + same parameters = **identical image**

**Why it matters**:
```python
# These will generate the SAME image:
image1 = adapter.generate(prompt="a cat", seed=42, width=1024)
image2 = adapter.generate(prompt="a cat", seed=42, width=1024)

# This will generate a DIFFERENT image:
image3 = adapter.generate(prompt="a cat", seed=123, width=1024)
```

**Use cases**:
- Iterate on a prompt while keeping the "pose/composition" the same
- Share reproducible results with others
- Debug generation issues

**In the code** (`core/adapters/zimage_turbo.py`):
```python
generator = torch.Generator(device).manual_seed(seed)
# Ensures deterministic noise initialization
```

### Plugin Hooks: Extending Generation

Pipeworks allows plugins to hook into the generation pipeline at 4 points:

```
1. on_generate_start()    → Modify parameters before generation
2. [GENERATION HAPPENS]
3. on_generate_complete() → Modify image after generation
4. on_before_save()       → Modify image/path before saving
5. [IMAGE SAVED]
6. on_after_save()        → Post-save actions (e.g., export metadata)
```

**Example**: The `Save Metadata` plugin uses `on_after_save()` to export a JSON file with all generation parameters alongside the image.

### Summary: Key Takeaways

1. **The model has 4 main components**: Tokenizer → Text Encoder → Diffusion Transformer → VAE Decoder
2. **Loading happens once** (lazy initialization), subsequent generations are fast
3. **Z-Image-Turbo is optimized for speed**: 9 steps, guidance_scale=0.0, sub-second generation
4. **Memory matters**: 16GB VRAM recommended, bfloat16 precision
5. **Seeds enable reproducibility**: Same seed = same image
6. **Plugins extend functionality**: Hook into generation at multiple lifecycle points

## Configuration

Configuration is managed via environment variables or `.env` file. All settings are prefixed with `PIPEWORKS_`.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPEWORKS_MODEL_ID` | `Tongyi-MAI/Z-Image-Turbo` | HuggingFace model ID |
| `PIPEWORKS_DEVICE` | `cuda` | Device for inference |
| `PIPEWORKS_TORCH_DTYPE` | `bfloat16` | Model precision |
| `PIPEWORKS_NUM_INFERENCE_STEPS` | `9` | Generation steps |
| `PIPEWORKS_GRADIO_SERVER_PORT` | `7860` | UI server port |

See `.env.example` for all available options.

## Development

### Project Structure

- **Core (`src/pipeworks/core/`)**: Model loading, inference, configuration, prompt building, gallery management
- **UI (`src/pipeworks/ui/`)**: Gradio interface with generation, gallery browser, and favorites
- **Plugins (`src/pipeworks/plugins/`)**: Extensibility system with lifecycle hooks
- **Workflows (`src/pipeworks/workflows/`)**: Content-specific generation strategies

### Implemented Features

- ✅ Plugin system for custom models and processors
- ✅ Workflow orchestration for content-specific generation (character, game assets, city maps)
- ✅ File-based prompt builder with multiple selection modes
- ✅ Gallery browser with filtering and favorites
- ✅ Tokenizer analysis for prompt optimization
- ✅ SQLite-based favorites database
- ✅ Catalog management for archiving images
- ✅ Metadata export plugin
- ✅ Comprehensive test suite (50%+ coverage, 93-100% for core logic)
- ✅ GitHub Actions CI/CD pipeline

### Future Roadmap

- [ ] Batch processing and queue management
- [ ] Image-to-image support
- [ ] ControlNet integration
- [ ] API server for remote access
- [ ] Additional model support (FLUX, SDXL, etc.)
- [ ] Inpainting and outpainting capabilities
- [ ] Fine-tuning utilities

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (when implemented)
pytest

# Format code
black src/
ruff check src/

# Type checking
mypy src/
```

## System Requirements

### Minimum
- Python 3.12+
- NVIDIA GPU with 16GB VRAM
- 50GB disk space
- 16GB RAM

### Recommended (tested configuration)
- AMD Ryzen 9 9900X (or equivalent)
- NVIDIA RTX 5090 (or RTX 4090)
- 64GB+ RAM
- 100GB+ SSD storage
- Debian Trixie / Ubuntu 22.04+ / similar Linux


## Code Quality & Testing

Pipeworks emphasizes maintainability and reliability:

### Testing
- **Test Coverage**: 50%+ overall, 93-100% for core business logic
- **Test Framework**: pytest with comprehensive fixtures
- **Test Organization**:
  - `tests/unit/`: Fast, isolated unit tests for individual components
  - `tests/integration/`: End-to-end tests for component interactions
- **Run tests**: `pytest` or `pytest -v` for verbose output

### Code Standards
- **Formatting**: Black (line length: 100 characters)
- **Linting**: Ruff (enforces PEP 8, modern Python patterns)
- **Type Hints**: Full type annotations throughout codebase
- **Target**: Python 3.12+ (modern syntax, type system)

### Continuous Integration
- **GitHub Actions** CI pipeline on `main` and `develop` branches
- **Checks**:
  - Test suite (Python 3.12 and 3.13)
  - Linting (ruff) and formatting (black)
  - Type checking (mypy, non-blocking)
  - Security scanning (Trivy, non-blocking)
  - Coverage reports (Codecov integration)

### Recent Refactoring (December 2025)

The codebase underwent a major refactoring to improve maintainability:

1. **Phase 1**: Extracted UI business logic from `app.py` (866 → 459 lines)
   - Separation of concerns: layout vs. business logic
   - Created dedicated modules: `handlers.py`, `formatting.py`, `adapters.py`, `validation.py`
   - All event handlers now independently testable

2. **Phase 2**: Comprehensive test suite
   - Added unit and integration tests
   - Achieved 93-100% coverage for core business logic
   - Set up GitHub Actions CI/CD

3. **Phase 3**: Code quality enforcement
   - Fixed all linting errors (ruff)
   - Full black formatting compliance
   - Type hints throughout

This refactoring demonstrates production-ready engineering practices and makes the codebase easy to extend and maintain.

## Acknowledgments

- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI - State-of-the-art 6B parameter diffusion model
- [Gradio](https://gradio.app/) - Elegant web UI framework for machine learning
- [Diffusers](https://github.com/huggingface/diffusers) - HuggingFace's diffusion model library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Pydantic](https://docs.pydantic.dev/) - Data validation and settings management

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow existing patterns, use Black and Ruff
2. **Testing**: Add tests for new features (aim for 80%+ coverage)
3. **Type Hints**: Include type annotations for all functions
4. **Documentation**: Update README and docstrings
5. **Commits**: Use clear, descriptive commit messages

See `CLAUDE.md` for detailed development guidelines.

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## License

[Add your license here]
# Test change
