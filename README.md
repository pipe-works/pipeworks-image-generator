# Pipeworks Image Generator

Programmatic image generation with Z-Image-Turbo and an extensible plugin system.

## Overview

Pipeworks is a Python-based image generation framework designed for programmatic control and experimentation. Moving beyond node-based interfaces like ComfyUI, Pipeworks provides a clean, code-first approach to image generation with a focus on extensibility and automation.

### Key Features

- **Z-Image-Turbo Integration**: Sub-second inference with the state-of-the-art 6B parameter model
- **Gradio UI**: Clean, responsive web interface for interactive generation
- **Programmatic API**: Full Python API for scripting and automation
- **Extensible Architecture**: Plugin system and workflow orchestration (coming soon)
- **Local-First**: Everything runs on your hardware, no cloud dependencies
- **Production-Ready**: Built with modern Python best practices

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

1. Open the web interface (default: `http://localhost:7860`)
2. Enter your prompt in the text box
3. Adjust generation parameters (width, height, steps, seed)
4. Click "Generate Image"
5. Generated images are saved to `outputs/` directory

### Programmatic Usage

```python
from pipeworks import ImageGenerator

# Initialize generator
generator = ImageGenerator()

# Generate image
image = generator.generate(
    prompt="A serene mountain landscape at sunset",
    width=1024,
    height=1024,
    seed=42,
)

# Generate and save
image, path = generator.generate_and_save(
    prompt="A cute cat sleeping on a cozy blanket",
    seed=12345,
)
print(f"Saved to: {path}")
```

## Architecture

```
pipeworks-image-generator/
├── src/pipeworks/
│   ├── core/              # Core generation engine
│   │   ├── config.py      # Configuration management
│   │   └── pipeline.py    # Z-Image-Turbo wrapper
│   ├── plugins/           # Plugin system (future)
│   ├── workflows/         # Workflow orchestration (future)
│   └── ui/                # Gradio interface
│       └── app.py
├── models/                # Model cache (gitignored)
├── outputs/               # Generated images (gitignored)
└── pyproject.toml         # Project configuration
```

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

- **Core (`src/pipeworks/core/`)**: Model loading, inference, configuration
- **UI (`src/pipeworks/ui/`)**: Gradio interface
- **Plugins (`src/pipeworks/plugins/`)**: Extensibility system (coming soon)
- **Workflows (`src/pipeworks/workflows/`)**: Automation pipelines (coming soon)

### Future Roadmap

- [ ] Plugin system for custom models and processors
- [ ] Workflow orchestration for multi-step generation
- [ ] Batch processing and queue management
- [ ] Image-to-image support
- [ ] ControlNet integration
- [ ] API server for remote access
- [ ] Additional model support (FLUX, SDXL, etc.)

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


## Acknowledgments

- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI
- [Gradio](https://gradio.app/) for the UI framework
- [Diffusers](https://github.com/huggingface/diffusers) for model integration

## Support

For issues, questions, or contributions, please open an issue on GitHub.
