# Pipeworks Image Generator

[![Test and Lint](https://github.com/pipe-works/pipeworks-image-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/pipe-works/pipeworks-image-generator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pipe-works/pipeworks-image-generator/branch/main/graph/badge.svg)](https://codecov.io/gh/pipe-works/pipeworks-image-generator)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Multi-model AI image generation with a FastAPI REST API and web frontend.

## Overview

Pipeworks Image Generator is a Python-based image generation system
that provides a FastAPI REST API backed by HuggingFace Diffusers
pipelines, with a vanilla HTML/CSS/JS frontend. It supports multiple
diffusion models, a three-part prompt composition system with fixed
boilerplate sections, a JSON-based gallery with favourites, and a
Ledgerfall pamphleteer aesthetic design system.

### Key Features

- **Multi-Model Support**: Load any HuggingFace text-to-image model via `AutoPipelineForText2Image`
- **FastAPI REST API**: Endpoints for generation, gallery management, prompt preview, runtime snippet source/auth, and statistics
- **Web Frontend**: Vanilla HTML/CSS/JS interface with the pipe-works design system — no build step
- **Three-Part Prompt System**: Compose prompts from prepend styles, scene descriptions, and
  append modifiers interleaved with fixed boilerplate
- **Canonical Policy Snippets**: Composer snippet dropdowns are sourced from canonical mud-server policy APIs
- **JSON Gallery**: Browse, filter, favourite, and delete generated images — no database required
- **Turbo Model Support**: Automatic guidance scale enforcement for turbo-distilled models
- **Deterministic Generation**: Seeded `torch.Generator` ensures same seed = same image
- **Self-Hosted Friendly**: Inference and policy APIs can run on your own machines/network
- **Comprehensive Tests**: 90 tests at 91%+ coverage

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 50GB+ free disk space (for model cache)

### Installation

```bash
git clone https://github.com/pipe-works/pipeworks-image-generator.git
cd pipeworks-image-generator
pip install -e .
```

`FLUX.2-klein-4B` requires a Diffusers build with `Flux2KleinPipeline`
support. At the moment that support may require installing Diffusers from
GitHub rather than PyPI:

```bash
pip install --upgrade "git+https://github.com/huggingface/diffusers.git"
```

### Configuration (Optional)

```bash
cp .env.example .env
# Edit .env with your preferences (device, dtype, port, etc.)
```

### Launch

```bash
pipeworks
```

The web UI will be available at `http://0.0.0.0:7860`.

## Usage

### Web Interface

1. Open the web UI in your browser
2. Select a model from the dropdown
3. Choose a prompt mode (manual text or automated preset)
4. Optionally select prepend/append style modifiers
5. Set dimensions, steps, guidance, and seed
6. Click Generate

Generated images appear in the gallery with full metadata. You can
favourite, filter, and delete images directly from the interface.

### Programmatic Usage

```python
from pipeworks.core.config import config
from pipeworks.core.model_manager import ModelManager

# Create manager and load a model
mgr = ModelManager(config)
mgr.load_model("Tongyi-MAI/Z-Image-Turbo")

# Generate an image
image = mgr.generate(
    prompt="A goblin repairing a clockwork automaton in a dimly lit workshop.",
    width=1024,
    height=1024,
    steps=4,
    guidance_scale=0.0,
    seed=42,
)

image.save("output.png")
mgr.unload()  # Free GPU memory
```

### REST API

All endpoints are documented in the FastAPI auto-generated docs at `/docs`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Web UI |
| GET | `/api/config` | Available models and prompt presets |
| POST | `/api/generate` | Generate image batch |
| POST | `/api/prompt/compile` | Preview compiled prompt |
| GET | `/api/runtime-mode` | Read snippet source mode (server dev/prod) |
| POST | `/api/runtime-mode` | Switch snippet source mode / URL |
| GET | `/api/runtime-auth` | Check runtime login/access for canonical policy APIs |
| POST | `/api/runtime-login` | Login to selected mud-server runtime |
| POST | `/api/runtime-logout` | Logout runtime session |
| GET | `/api/policy-prompts` | Load canonical policy snippet options/groups |
| GET | `/api/gallery` | Paginated gallery listing |
| GET | `/api/gallery/{id}` | Single gallery entry |
| GET | `/api/gallery/{id}/prompt` | Prompt metadata |
| POST | `/api/gallery/favourite` | Toggle favourite status |
| DELETE | `/api/gallery/{id}` | Delete image |
| GET | `/api/stats` | Gallery statistics |

## Architecture

```text
src/pipeworks/
├── core/                  # Generation engine (no HTTP dependency)
│   ├── config.py          # Pydantic Settings (PIPEWORKS_* env vars)
│   └── model_manager.py   # Diffusers pipeline lifecycle
├── api/                   # FastAPI REST API
│   ├── main.py            # App, routes, CLI entry point
│   ├── models.py          # Pydantic request models
│   └── prompt_builder.py  # Three-part prompt compilation
├── static/                # CSS, JS, fonts, data, gallery images
└── templates/             # index.html
```

### Design Decisions

- **FastAPI over Gradio**: Direct control over API design, static file serving, and frontend customisation
- **JSON over SQLite**: Gallery persistence in a single `gallery.json` file — simple, portable, no migrations
- **ModelManager over Adapter Registry**: One pipeline in memory at a time with explicit model switching
- **Vanilla JS over React/Vue**: No build step, instant page loads, full control over the design system
- **Lifespan over on_event**: Modern FastAPI lifecycle pattern for startup/shutdown

## Configuration

Most settings use `PIPEWORKS_*` environment variables. Runtime snippet source
settings use `PW_POLICY_*` variables. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `PIPEWORKS_DEVICE` | `cuda` | Compute device (cuda, mps, cpu) |
| `PIPEWORKS_TORCH_DTYPE` | `bfloat16` | Model precision |
| `PIPEWORKS_NUM_INFERENCE_STEPS` | `9` | Default inference steps |
| `PIPEWORKS_SERVER_HOST` | `0.0.0.0` | Server bind address |
| `PIPEWORKS_SERVER_PORT` | `7860` | Server port |
| `PIPEWORKS_DISABLE_HTTP_CACHE` | `false` | Disable browser caching for local frontend testing |
| `PIPEWORKS_MODELS_DIR` | `models` | Model cache directory |
| `PW_POLICY_SOURCE_MODE` | `server_dev` | Active snippet source mode (`server_dev`, `server_prod`) |
| `PW_POLICY_DEV_MUD_API_BASE_URL` | `http://127.0.0.1:8000` | Canonical policy API URL for dev mode |
| `PW_POLICY_PROD_MUD_API_BASE_URL` | `https://mud-api.example.com` | Canonical policy API URL for prod mode |

Examples:

```bash
# Local dev mud server
PW_POLICY_SOURCE_MODE=server_dev
PW_POLICY_DEV_MUD_API_BASE_URL=http://127.0.0.1:8000
PW_POLICY_PROD_MUD_API_BASE_URL=https://mud-api.example.com
```

```bash
# Shared/staging mud server
PW_POLICY_SOURCE_MODE=server_prod
PW_POLICY_DEV_MUD_API_BASE_URL=https://mud-dev.example.com
PW_POLICY_PROD_MUD_API_BASE_URL=https://mud-api.example.com
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
black src/ tests/
```

### Test Suite

- **90 tests** across 6 test files
- **91%+ coverage** on core and API packages
- Unit tests for config, model manager, prompt builder, Pydantic models
- Integration tests for all 10 API endpoints via FastAPI TestClient

## System Requirements

### Minimum

- Python 3.12+
- NVIDIA GPU with 16GB VRAM
- 50GB disk space
- 16GB RAM

### Recommended

- NVIDIA RTX 4090/5090 or equivalent
- 64GB+ RAM
- 100GB+ SSD storage

## Acknowledgments

- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI
- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) by Stability AI
- [SDXL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) by Stability AI
- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) by Black Forest Labs
- [Diffusers](https://github.com/huggingface/diffusers) by HuggingFace
- [FastAPI](https://fastapi.tiangolo.com/) by Sebastián Ramírez
- [PyTorch](https://pytorch.org/)
- [Pydantic](https://docs.pydantic.dev/)

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.
