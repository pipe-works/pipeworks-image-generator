# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pipeworks is a Python-based image generation framework for Z-Image-Turbo that provides both a programmatic API and a Gradio web UI. The project emphasizes code-first design over node-based interfaces, with a focus on extensibility through plugins and workflows.

**Key Technologies:**
- Python 3.12+ (type hints, modern syntax)
- Z-Image-Turbo (6B parameter model via HuggingFace Diffusers)
- Gradio 5.0+ for web UI
- Pydantic for configuration and data validation
- PyTorch for model inference

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
# Launch Gradio UI (preferred)
pipeworks

# Direct module execution
python -m pipeworks.ui.app
```

The UI will be accessible at `http://0.0.0.0:7860` by default.

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run tests matching pattern
pytest -k "test_validation"

# Run unit tests only (fast)
pytest tests/unit/ -v

# Run integration tests only (may be slow)
pytest tests/integration/ -v

# Run with detailed output
pytest -vv --showlocals
```

**Test Coverage:** The project maintains 50%+ overall coverage, with core business logic (models, validation, state, components) at 93-100% coverage. The UI glue code in `app.py` is excluded from coverage as it's difficult to unit test.

### Code Quality
```bash
# Check linting (ruff)
ruff check src/

# Auto-fix linting issues
ruff check src/ --fix

# Format code (black)
black src/

# Check formatting without changes
black --check src/

# Type checking (mypy)
mypy src/pipeworks/ui/ --ignore-missing-imports
```

**Code Standards:**
- Line length: 100 characters
- Target version: Python 3.12
- Black formatter for consistent style
- Ruff for linting (rules: E, F, I, N, W, UP)

### CI/CD
The project uses GitHub Actions for CI on `main` and `develop` branches:
- Test suite runs on Python 3.12 and 3.13
- Linting and formatting checks
- Type checking with mypy (non-blocking)
- Security scanning with Trivy (non-blocking)
- Coverage reports to Codecov

## Architecture

### High-Level Structure

```
src/pipeworks/
├── core/              # Core generation engine
│   ├── config.py      # Pydantic configuration (env-based)
│   ├── model_adapters.py  # Multi-model adapter system + registry
│   ├── adapters/      # Model-specific implementations
│   │   ├── zimage_turbo.py  # Z-Image-Turbo text-to-image adapter
│   │   └── qwen_image_edit.py # Qwen-Image-Edit editing adapter
│   ├── prompt_builder.py  # File-based prompt construction
│   ├── tokenizer.py   # Tokenization analysis utilities
│   ├── character_conditions.py  # Procedural character generation
│   ├── facial_conditions.py     # Facial signal generation
│   ├── gallery_browser.py  # Gallery browsing and filtering
│   ├── favorites_db.py     # SQLite favorites database
│   └── catalog_manager.py  # Archive management
├── plugins/           # Plugin system
│   ├── base.py        # PluginBase + PluginRegistry
│   └── save_metadata.py  # Built-in metadata export plugin
├── workflows/         # Workflow system
│   ├── base.py        # WorkflowBase + WorkflowRegistry
│   ├── character.py   # Character generation workflow
│   ├── game_asset.py  # Game asset workflow
│   └── city_map.py    # City/map generation workflow
└── ui/                # Gradio web interface
    ├── app.py         # Main UI layout and event wiring
    ├── components.py  # Reusable Gradio component builders
    ├── handlers/      # Event handler business logic (5 modules)
    ├── models.py      # Pydantic models for UI state
    ├── state.py       # UI state management
    ├── validation.py  # Input validation logic
    ├── formatting.py  # Output formatting utilities
    └── adapters.py    # UI value conversion functions
```

### Key Architectural Patterns

#### 1. Configuration System (`core/config.py`)
- Single `PipeworksConfig` class using Pydantic Settings
- All settings loaded from environment variables (prefix: `PIPEWORKS_`)
- Global `config` instance available via `from pipeworks.core.config import config`
- Environment variables override defaults (see `.env.example`)

#### 2. Plugin Architecture (`plugins/base.py`)
Plugins hook into the generation lifecycle at four points:
- `on_generate_start(params)` - Modify parameters before generation
- `on_generate_complete(image, params)` - Modify image after generation
- `on_before_save(image, path, params)` - Modify image/path before saving
- `on_after_save(image, path, params)` - Post-save actions (e.g., metadata export)

Plugins are registered globally via `plugin_registry` and instantiated per-session in the UI.

**Example Plugin Usage:**
```python
from pipeworks.plugins.base import plugin_registry

# Get available plugins
available = plugin_registry.list_available()

# Instantiate a plugin
plugin = plugin_registry.instantiate("Save Metadata", output_folder="metadata")
```

#### 3. Workflow System (`workflows/base.py`)
Workflows encapsulate generation strategies for specific content types (characters, game assets, maps). Each workflow defines:
- Prompt engineering approach (`build_prompt()`)
- Default parameters
- Pre/post processing hooks
- UI controls specific to the workflow

Workflows are registered via `workflow_registry` (similar pattern to plugins).

#### 4. UI State Management (`ui/state.py`, `ui/models.py`)
- Session-based state using Gradio's `gr.State`
- `UIState` Pydantic model contains all session data (prompt builder, tokenizer, generation params)
- State is lazily initialized via `initialize_ui_state()` to avoid loading heavy resources upfront
- Components pass state between handlers for persistence

**Important:** UI handlers must return the updated state as part of their outputs to maintain state consistency.

#### 5. Prompt Builder (`core/prompt_builder.py`)
Constructs prompts by combining text and random lines from files in the `inputs/` directory. Supports:
- File browsing with folder navigation
- Multiple selection modes: random line, specific line, line range, all lines, N random lines
- Three segments (start, middle, end) for flexible composition
- File content caching for performance

#### 6. Refactored UI Architecture (`ui/`)
The UI was recently refactored (commit 39b5b1a) to separate concerns:
- **app.py**: Only UI layout, Gradio components, and event wiring
- **handlers.py**: All event handler business logic (generation, navigation, validation)
- **components.py**: Reusable UI component builders (SegmentUI class)
- **formatting.py**: Output formatting for info/error messages
- **adapters.py**: Conversion between UI values and domain objects
- **models.py**: Pydantic data models (GenerationParams, SegmentConfig, UIState)
- **validation.py**: Input validation with custom ValidationError
- **state.py**: State initialization logic

This separation makes the codebase more testable and maintainable.

### Critical Implementation Details

#### Model Adapter Lifecycle
```python
from pipeworks import model_registry, config

# Initialize adapter (doesn't load model yet)
adapter = model_registry.instantiate("Z-Image-Turbo", config)

# Load model (lazy loading)
adapter.load_model()  # Called automatically on first generate()

# Generate image
image = adapter.generate(
    prompt="...",
    width=1024,
    height=1024,
    num_inference_steps=9,
    seed=42
)

# Generate and save (runs plugin hooks)
image, path = adapter.generate_and_save(prompt="...", seed=42)

# Cleanup
adapter.unload_model()
```

**Plugin Injection:**
```python
# Pass plugins during initialization
from pipeworks import model_registry, config
from pipeworks.plugins.base import plugin_registry

metadata_plugin = plugin_registry.instantiate("Save Metadata")
adapter = model_registry.instantiate("Z-Image-Turbo", config, plugins=[metadata_plugin])
```

#### Z-Image-Turbo Specifics
- **Guidance Scale**: Must be 0.0 for Turbo models (enforced in adapters/zimage_turbo.py)
- **Optimal Steps**: 9 inference steps (config default)
- **Dtype**: bfloat16 recommended (config default)
- **Device**: cuda preferred, falls back to cpu

#### UI Component Reuse (`ui/components.py`)
The `SegmentUI` class eliminates code duplication for the three prompt builder segments:
```python
# Creates identical UI controls for start/middle/end segments
start_segment, middle_segment, end_segment = create_three_segments(initial_choices)

# Access components
file_dropdown = segment.file
mode_dropdown = segment.mode

# Get all inputs for event handlers
inputs = segment.get_input_components()
```

#### Session State Pattern
```python
# Handler function signature
def handle_event(input_value: str, state: UIState) -> tuple[str, UIState]:
    # Initialize state if needed
    state = initialize_ui_state(state)

    # Access state components
    state.prompt_builder.do_something()

    # Update state
    state.last_seed = 42

    # Return outputs including updated state
    return output_value, state
```

## Common Development Patterns

### Adding a New Plugin

1. **Create plugin class** in `src/pipeworks/plugins/`:
```python
from pipeworks.plugins.base import PluginBase, plugin_registry

class MyPlugin(PluginBase):
    name = "My Plugin"
    description = "Does something cool"
    version = "0.1.0"

    def on_after_save(self, image, path, params):
        # Your plugin logic
        pass

# Register the plugin
plugin_registry.register(MyPlugin)
```

2. **Import in** `src/pipeworks/plugins/__init__.py`:
```python
from .my_plugin import MyPlugin
```

3. **Add UI controls** in `src/pipeworks/ui/app.py` (plugins section)

### Adding a New Workflow

1. **Create workflow class** in `src/pipeworks/workflows/`:
```python
from pipeworks.workflows.base import WorkflowBase, workflow_registry

class MyWorkflow(WorkflowBase):
    name = "My Workflow"
    description = "Generate specific content type"
    version = "0.1.0"

    def build_prompt(self, **kwargs) -> str:
        # Build workflow-specific prompt
        return f"Your prompt template with {kwargs['param']}"

# Register the workflow
workflow_registry.register(MyWorkflow)
```

2. **Import in** `src/pipeworks/workflows/__init__.py`

### Writing Tests

Follow the established testing patterns in `tests/`:

**Unit tests** (`tests/unit/`):
- Test individual functions/classes in isolation
- Use fixtures from `conftest.py` (temp_dir, test_config, etc.)
- Mock external dependencies (file I/O, model loading)
- Focus on business logic validation

**Integration tests** (`tests/integration/`):
- Test component interactions
- Use real file systems (via temp_dir fixture)
- Test end-to-end workflows

**Example test structure:**
```python
def test_something(test_inputs_dir, valid_segment_config):
    """Test description."""
    # Arrange
    builder = PromptBuilder(test_inputs_dir)

    # Act
    result = builder.do_something(valid_segment_config)

    # Assert
    assert result == expected_value
```

### Adding Configuration Options

1. **Add to** `PipeworksConfig` in `src/pipeworks/core/config.py`:
```python
class PipeworksConfig(BaseSettings):
    my_setting: str = Field(
        default="default_value",
        description="Setting description"
    )
```

2. **Update** `.env.example` with the new variable:
```bash
PIPEWORKS_MY_SETTING=default_value
```

3. **Access via** `from pipeworks.core.config import config; config.my_setting`

## Important Constraints

1. **Z-Image-Turbo requires guidance_scale=0.0** - This is enforced in adapters/zimage_turbo.py with a warning
2. **Models directory is large** (50GB+) - Always in `.gitignore`
3. **UI state must be returned** from handlers to maintain session consistency
4. **Plugin hooks are optional** - Not all plugins need to implement all hooks
5. **Prompt builder caches file contents** - Call `clear_cache()` if files change during runtime
6. **Type hints are required** - Project uses modern Python typing throughout
7. **All configuration via environment variables** - No hardcoded paths or credentials

## Environment Variables

See `.env.example` for all available settings. Key variables:

- `PIPEWORKS_MODEL_ID`: HuggingFace model ID
- `PIPEWORKS_DEVICE`: cuda, mps, or cpu
- `PIPEWORKS_TORCH_DTYPE`: bfloat16, float16, or float32
- `PIPEWORKS_NUM_INFERENCE_STEPS`: Default inference steps (9 recommended)
- `PIPEWORKS_MODELS_DIR`: Model cache location
- `PIPEWORKS_OUTPUTS_DIR`: Generated images location
- `PIPEWORKS_INPUTS_DIR`: Prompt builder text files location
- `PIPEWORKS_GRADIO_SERVER_PORT`: UI server port (default: 7860)

## Recent Refactoring (December 2025)

The codebase recently underwent a major refactoring to improve maintainability and testability:

1. **Phase 1** (commit 39b5b1a): Extracted UI business logic from `app.py` (866 → 459 lines)
   - Created `handlers.py`, `formatting.py`, `adapters.py`
   - All event handlers now separately testable

2. **Testing & CI** (commit 7741739): Added comprehensive test suite
   - 93-100% coverage for core business logic
   - GitHub Actions CI with Python 3.12/3.13

3. **Code Quality** (commit 979f045): Fixed all linting errors
   - Full black formatting compliance
   - All ruff checks passing

When modifying the UI, maintain the separation of concerns established in this refactoring.
