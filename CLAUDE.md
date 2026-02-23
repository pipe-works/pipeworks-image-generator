# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pipe-Works Image Generator is a Python-based image generation system
with a FastAPI REST API backend and a vanilla HTML/CSS/JS frontend.
It supports multiple diffusion models via HuggingFace Diffusers with
real inference, a JSON-based gallery, and a three-part prompt
composition system.

**Key Technologies:**

- Python 3.12+ (type hints, modern syntax)
- FastAPI + Uvicorn (REST API and static file serving)
- HuggingFace Diffusers (multi-model pipeline support)
- Pydantic / Pydantic Settings (configuration and request validation)
- PyTorch (model inference)
- Vanilla HTML/CSS/JS frontend (no build step)

## Development Commands

### Installation

```bash
# Development install with all dependencies
pip install -e ".[dev]"

# Production install only
pip install -e .
```

### Running the Application

```bash
# Launch FastAPI server (installed entry point)
pipeworks

# Direct module execution
python -m pipeworks.api.main
```

The server will be accessible at `http://0.0.0.0:7860` by default.

### Testing

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_config.py

# Run tests matching pattern
pytest -k "test_turbo"

# Run unit tests only (fast)
pytest tests/unit/ -v

# Run integration tests only (FastAPI TestClient)
pytest tests/integration/ -v

# Run without coverage (faster)
pytest --no-cov
```

**Test Coverage:** 90 tests, 91%+ overall coverage. Core config and
prompt builder at 100%, API routes at 91%, model manager at 87%.

### Code Quality

```bash
# Check linting (ruff)
ruff check src/ tests/

# Auto-fix linting issues
ruff check src/ tests/ --fix

# Format code (black)
black src/ tests/

# Check formatting without changes
black --check src/ tests/
```

**Code Standards:**

- Line length: 100 characters
- Target version: Python 3.12
- Black formatter for consistent style
- Ruff for linting (rules: E, F, I, N, W, UP)
- **Pre-commit hooks enabled** — auto-fixes on commit

## Architecture

### High-Level Structure

```text
src/pipeworks/
├── __init__.py            # Package root, re-exports ModelManager + config
├── core/                  # Core generation engine (no HTTP dependency)
│   ├── __init__.py        # Re-exports ModelManager, PipeworksConfig, config
│   ├── config.py          # Pydantic Settings (PIPEWORKS_* env prefix)
│   └── model_manager.py   # Diffusers pipeline lifecycle management
├── api/                   # FastAPI REST API layer
│   ├── __init__.py        # Module docstring
│   ├── main.py            # FastAPI app, all routes, CLI entry point
│   ├── models.py          # Pydantic request models (GenerateRequest, etc.)
│   └── prompt_builder.py  # Three-part prompt template compilation
├── static/                # Web-accessible static assets (served at /static/)
│   ├── css/               # Stylesheets (app.css, pipe-works-base.css, fonts)
│   ├── js/                # Frontend JavaScript (app.js)
│   ├── data/              # models.json, prompts.json, gallery.json
│   ├── fonts/             # Woff2 font files (16 files)
│   └── gallery/           # Generated images (gitignored, auto-created)
└── templates/             # HTML templates
    └── index.html         # Main application page
```

### Key Architectural Patterns

#### 1. Configuration System (`core/config.py`)

- Single `PipeworksConfig` class using Pydantic Settings
- All settings loaded from environment variables (prefix: `PIPEWORKS_`)
- Global `config` instance: `from pipeworks.core.config import config`
- Auto-creates `models_dir`, `outputs_dir`, `gallery_dir` on init
- Path defaults resolve to package-internal `static/` and `templates/` directories

#### 2. Model Manager (`core/model_manager.py`)

- `ModelManager` manages one diffusers pipeline at a time
- Lazy loading — model loads on first `generate()` or explicit `load_model()`
- Model switching — unloads current model, clears CUDA, loads new one
- Turbo enforcement — models with "turbo" in their HF ID get `guidance_scale=0.0`
- Deterministic seeding — `torch.Generator(device).manual_seed(seed)` per call
- Performance options — attention slicing, CPU offload, torch.compile (from config)

#### 3. API Layer (`api/main.py`)

FastAPI application with 10 REST endpoints:

| Method   | Path                        | Purpose                           |
|----------|-----------------------------|-----------------------------------|
| GET      | `/`                         | Serve HTML page                   |
| GET      | `/api/config`               | Models, prompts, aspect ratios    |
| POST     | `/api/generate`             | Generate image batch              |
| POST     | `/api/prompt/compile`       | Preview compiled prompt           |
| GET      | `/api/gallery`              | Paginated gallery with filters    |
| GET      | `/api/gallery/{id}`         | Single gallery entry              |
| GET      | `/api/gallery/{id}/prompt`  | Prompt metadata for an image      |
| POST     | `/api/gallery/favourite`    | Toggle favourite status           |
| DELETE   | `/api/gallery/{id}`         | Delete image + metadata           |
| GET      | `/api/stats`                | Gallery totals and per-model counts |

- Lifespan context manager for ModelManager setup/teardown
- JSON-based gallery persistence (no database)
- StaticFiles mount for CSS/JS/fonts/gallery images
- HTMLResponse for index.html (no template engine needed)

#### 4. Three-Part Prompt System (`api/prompt_builder.py`)

Prompts are compiled from three user-selectable parts interleaved with fixed boilerplate:

```text
[Prepend Style]  (optional)
[Fixed: Ledgerfall pamphleteer aesthetic]
Main Scene:
[Manual Prompt or Automated Preset]
[Fixed: Mood/atmosphere]
[Append Modifier]  (optional)
[Fixed: Colour palette directive]
```

#### 5. Frontend

- Vanilla HTML/CSS/JS — no build step, no framework
- Uses pipe-works design system tokens (`pipe-works-base.css`)
- Fetches config dynamically via `GET /api/config` on page load
- All interactions through REST API calls

### Critical Implementation Details

#### ModelManager Lifecycle

```python
from pipeworks.core.config import config
from pipeworks.core.model_manager import ModelManager

mgr = ModelManager(config)
mgr.load_model("Tongyi-MAI/Z-Image-Turbo")

image = mgr.generate(
    prompt="a goblin workshop",
    width=1024, height=1024,
    steps=4, guidance_scale=0.0, seed=42,
)

mgr.unload()  # Frees GPU memory
```

#### Turbo Model Constraints

- **Guidance Scale**: Must be 0.0 (enforced automatically in `model_manager.py`)
- **Optimal Steps**: 4-9 inference steps
- **Dtype**: bfloat16 recommended (config default)
- **Device**: cuda preferred, falls back to cpu

#### Gallery Persistence

- Single `gallery.json` file in `static/data/`
- Self-bootstrapping — created on first generation
- Newest images inserted at position 0 (reverse chronological)
- Image PNGs stored in `static/gallery/` (web-accessible)

### Writing Tests

**Unit tests** (`tests/unit/`):

- `test_config.py` — Config defaults, validation, directory creation
- `test_model_manager.py` — Loading, generation, turbo, unload (all mocked)
- `test_prompt_builder.py` — Prompt compilation, empty parts, boilerplate
- `test_api_models.py` — Pydantic validation, defaults, serialisation

**Integration tests** (`tests/integration/`):

- `test_api.py` — All 10 endpoints via FastAPI TestClient with mocked ModelManager

**Test patterns:**

- `conftest.py` provides `test_config`, `test_client`, `mock_model_manager`, `sample_gallery`
- ModelManager mocking uses `sys.modules` injection for lazy torch/diffusers imports
- API tests use `unittest.mock.patch` on module-level path constants

### Adding Configuration Options

1. Add field to `PipeworksConfig` in `src/pipeworks/core/config.py`
2. Update `.env.example` with the new `PIPEWORKS_*` variable
3. Access via `from pipeworks.core.config import config; config.my_setting`

## Important Constraints

1. **Turbo models require guidance_scale=0.0** — enforced in model_manager.py
2. **Models directory is large** (50GB+) — always in `.gitignore`
3. **No template engine** — HTMLResponse serves static index.html
4. **Gallery is JSON-based** — no SQLite or database dependency
5. **Type hints are required** — project uses modern Python typing throughout
6. **All configuration via environment variables** — no hardcoded paths or credentials
7. **Detailed comments and docstrings required** — all code must be thoroughly documented

## Environment Variables

See `.env.example` for all available settings. Key variables:

| Variable | Default | Description |
|---|---|---|
| `PIPEWORKS_DEVICE` | `cuda` | Compute device (cuda, mps, cpu) |
| `PIPEWORKS_TORCH_DTYPE` | `bfloat16` | Model precision |
| `PIPEWORKS_NUM_INFERENCE_STEPS` | `9` | Default inference steps |
| `PIPEWORKS_GUIDANCE_SCALE` | `0.0` | Default guidance (0.0 for turbo) |
| `PIPEWORKS_SERVER_HOST` | `0.0.0.0` | Server bind address |
| `PIPEWORKS_SERVER_PORT` | `7860` | Server port |
| `PIPEWORKS_MODELS_DIR` | `models` | Model cache location |
| `PIPEWORKS_OUTPUTS_DIR` | `outputs` | Generated images directory |
