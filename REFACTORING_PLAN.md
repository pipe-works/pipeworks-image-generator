# Refactoring Plan: Critical Issues Fix
**Branch:** `refactor/fix-critical-issues`
**Date:** December 15, 2025
**Based on:** gradio_analysis_report.md

---

## Executive Summary

This refactoring addresses the critical architectural issues identified in the Gradio analysis report. The primary goals are:

1. **Thread-safe state management** - Replace global state with session-based state
2. **Maintainable parameter passing** - Use dataclasses to reduce 37-parameter functions
3. **Code deduplication** - Create reusable segment components
4. **Robust error handling** - Add validation and proper error propagation
5. **Test coverage** - Establish test infrastructure

**Estimated Total Effort:** 5-7 days of development + testing

---

## Phase 1: Foundation - Dataclasses & Type Safety (Priority: CRITICAL)

### 1.1 Create Data Models

**File:** `src/pipeworks/ui/models.py` (new)

```python
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class SegmentConfig:
    """Configuration for a single prompt segment (start/middle/end)."""
    text: str = ""
    path: str = ""
    file: str = "(None)"
    mode: str = "Random Line"
    line: int = 1
    range_end: int = 1
    count: int = 1
    dynamic: bool = False

    def is_configured(self) -> bool:
        """Check if segment has a valid file selected."""
        return self.file and self.file != "(None)" and not self.file.startswith("📁")

@dataclass
class GenerationParams:
    """Parameters for image generation."""
    prompt: str
    width: int
    height: int
    num_steps: int
    batch_size: int
    runs: int
    seed: int
    use_random_seed: bool

    def validate(self) -> None:
        """Validate generation parameters."""
        if self.batch_size < 1 or self.batch_size > 100:
            raise ValueError(f"Batch size must be 1-100, got {self.batch_size}")
        if self.runs < 1 or self.runs > 100:
            raise ValueError(f"Runs must be 1-100, got {self.runs}")
        if self.batch_size * self.runs > 1000:
            raise ValueError(f"Total images ({self.batch_size * self.runs}) exceeds limit of 1000")
        if self.width % 64 != 0:
            raise ValueError(f"Width must be multiple of 64, got {self.width}")
        if self.height % 64 != 0:
            raise ValueError(f"Height must be multiple of 64, got {self.height}")
        if self.width < 512 or self.width > 2048:
            raise ValueError(f"Width must be 512-2048, got {self.width}")
        if self.height < 512 or self.height > 2048:
            raise ValueError(f"Height must be 512-2048, got {self.height}")

@dataclass
class UIState:
    """Session state for the UI."""
    generator: Optional[object] = None  # ImageGenerator instance
    tokenizer_analyzer: Optional[object] = None  # TokenizerAnalyzer instance
    prompt_builder: Optional[object] = None  # PromptBuilder instance
    active_plugins: dict = None  # Dict[str, PluginBase]

    def __post_init__(self):
        if self.active_plugins is None:
            self.active_plugins = {}
```

**Benefits:**
- Reduces `generate_image()` from 37 parameters to 4 clean parameters
- Built-in validation logic
- Type safety and IDE autocomplete
- Easy to test and maintain

**Effort:** 1 day

---

## Phase 2: Session-Based State Management (Priority: CRITICAL)

### 2.1 Replace Global State with Gradio State

**Current Problem:**
```python
# Lines 24-28 - Global state (UNSAFE!)
active_plugins: Dict[str, PluginBase] = {}
generator: ImageGenerator = None
tokenizer_analyzer: TokenizerAnalyzer = None
prompt_builder: PromptBuilder = None
```

**Solution:**
```python
# Use Gradio's gr.State() for session isolation
def create_ui():
    with gr.Blocks() as app:
        # Create session state - one per user
        ui_state = gr.State(UIState())

        # Initialize on first use
        def initialize_state(state: UIState) -> UIState:
            if state.generator is None:
                state.generator = ImageGenerator(config, plugins=[])
                state.tokenizer_analyzer = TokenizerAnalyzer(config.model_id, config.models_dir)
                state.prompt_builder = PromptBuilder(config.inputs_dir)
                state.generator.load_model()  # Pre-load
            return state

        # All callbacks receive and return state
        def generate_image_safe(
            params: GenerationParams,
            segments: tuple[SegmentConfig, SegmentConfig, SegmentConfig],
            state: UIState
        ) -> tuple[List[str], str, str, UIState]:
            state = initialize_state(state)
            # Use state.generator instead of global generator
            # ... generation logic ...
            return images, info, seed, state  # Return updated state
```

**Key Changes:**
1. Remove all `global` keywords (8 instances)
2. Add `ui_state = gr.State(UIState())` in `create_ui()`
3. Update all event handlers to accept and return `state`
4. Lazy initialize per session on first use

**Benefits:**
- Thread-safe for concurrent users
- Each user has isolated state
- No race conditions
- Testable with explicit state passing

**Effort:** 2 days

---

## Phase 3: Code Deduplication - Reusable Components (Priority: HIGH)

### 3.1 Create SegmentUI Class

**File:** `src/pipeworks/ui/components.py` (new)

```python
from typing import List, Tuple
import gradio as gr
from .models import SegmentConfig

class SegmentUI:
    """Reusable UI component for a prompt segment."""

    def __init__(self, name: str, initial_choices: List[str]):
        self.name = name

        with gr.Group():
            self.title = gr.Markdown(f"**{name} Segment**")
            self.text = gr.Textbox(label=f"{name} Text", placeholder="Optional text...", lines=1)
            self.path_display = gr.Textbox(label="Current Path", value="/inputs", interactive=False)
            self.file = gr.Dropdown(label="File/Folder Browser", choices=initial_choices, value="(None)")
            self.path_state = gr.State(value="")

            with gr.Row():
                self.mode = gr.Dropdown(
                    label="Mode",
                    choices=["Random Line", "Specific Line", "Line Range", "All Lines", "Random Multiple"],
                    value="Random Line"
                )
                self.dynamic = gr.Checkbox(label="Dynamic", value=False, info="Rebuild this segment for each image")

            with gr.Row():
                self.line = gr.Number(label="Line #", value=1, minimum=1, precision=0, visible=False)
                self.range_end = gr.Number(label="End Line #", value=1, minimum=1, precision=0, visible=False)
                self.count = gr.Number(label="Count", value=1, minimum=1, maximum=10, precision=0, visible=False)

    def get_inputs(self) -> List[gr.components.Component]:
        """Return all input components for this segment."""
        return [
            self.text, self.path_state, self.file, self.mode,
            self.line, self.range_end, self.count, self.dynamic
        ]

    def get_outputs(self) -> List[gr.components.Component]:
        """Return all output components for this segment."""
        return [self.title, self.path_display, self.file, self.path_state]

    def to_config(self, *values) -> SegmentConfig:
        """Convert UI values to SegmentConfig dataclass."""
        text, path, file, mode, line, range_end, count, dynamic = values
        return SegmentConfig(
            text=text, path=path, file=file, mode=mode,
            line=line, range_end=range_end, count=count, dynamic=dynamic
        )
```

**Usage:**
```python
# Instead of 3× identical blocks of 150+ lines...
start = SegmentUI("Start", initial_choices)
middle = SegmentUI("Middle", initial_choices)
end = SegmentUI("End", initial_choices)

# Collect all inputs easily
all_segment_inputs = start.get_inputs() + middle.get_inputs() + end.get_inputs()
```

**Benefits:**
- Eliminates 400+ lines of duplicated code
- Single source of truth for segment UI
- Bug fixes apply to all segments automatically
- Easy to add new segments if needed

**Effort:** 1 day

---

## Phase 4: Error Handling & Validation (Priority: HIGH)

### 4.1 Add Input Validation

**File:** `src/pipeworks/ui/validation.py` (new)

```python
from pathlib import Path
from typing import Tuple
from .models import GenerationParams, SegmentConfig

class ValidationError(Exception):
    """User-friendly validation error."""
    pass

def validate_generation_params(params: GenerationParams) -> None:
    """
    Validate generation parameters with user-friendly messages.

    Raises:
        ValidationError: If validation fails with user-friendly message
    """
    try:
        params.validate()  # Uses dataclass validation
    except ValueError as e:
        raise ValidationError(str(e))

def validate_segment_path(path: str, file: str, base_dir: Path) -> Path:
    """
    Validate that a segment file path is safe and exists.

    Args:
        path: Relative path from base_dir
        file: Filename
        base_dir: Base directory (inputs_dir)

    Returns:
        Absolute resolved path

    Raises:
        ValidationError: If path is invalid or unsafe
    """
    if not file or file == "(None)" or file.startswith("📁"):
        raise ValidationError("No file selected")

    # Build and resolve path
    full_path = (base_dir / path / file).resolve()

    # Security: Ensure path is within base_dir (prevent path traversal)
    if not str(full_path).startswith(str(base_dir.resolve())):
        raise ValidationError(f"Invalid path: {full_path}")

    # Check file exists
    if not full_path.exists():
        raise ValidationError(f"File not found: {file}")

    return full_path

def validate_segments(
    segments: Tuple[SegmentConfig, SegmentConfig, SegmentConfig],
    base_dir: Path
) -> None:
    """Validate all segment configurations."""
    for i, segment in enumerate(segments):
        if segment.is_configured():
            try:
                validate_segment_path(segment.path, segment.file, base_dir)
            except ValidationError as e:
                segment_name = ["Start", "Middle", "End"][i]
                raise ValidationError(f"{segment_name} segment: {e}")
```

### 4.2 Update Error Handling in generate_image()

```python
def generate_image(
    params: GenerationParams,
    segments: Tuple[SegmentConfig, SegmentConfig, SegmentConfig],
    state: UIState
) -> Tuple[List[str], str, str, UIState]:
    """Generate images with proper error handling."""
    try:
        # Initialize state
        state = initialize_state(state)

        # Validate inputs
        validate_generation_params(params)
        validate_segments(segments, config.inputs_dir)

        # ... generation logic ...

        return generated_paths, info, str(seeds_used[-1]), state

    except ValidationError as e:
        # User-friendly error
        logger.warning(f"Validation error: {e}")
        return [], f"❌ **Validation Error:** {str(e)}", str(params.seed), state

    except Exception as e:
        # Unexpected error
        logger.error(f"Generation error: {e}", exc_info=True)
        return [], f"❌ **Error:** An unexpected error occurred. Check logs for details.", str(params.seed), state
```

**Benefits:**
- No more silent failures
- User-friendly error messages with emoji
- Security validation (path traversal prevention)
- Proper error logging
- Errors don't become prompts!

**Effort:** 1 day

---

## Phase 5: Testing Infrastructure (Priority: HIGH)

### 5.1 Set Up Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── test_models.py              # Test dataclasses
├── test_validation.py          # Test validation logic
├── test_prompt_builder.py      # Test prompt building
├── test_ui_components.py       # Test SegmentUI
└── test_generation.py          # Test generation flow
```

### 5.2 Sample Tests

**File:** `tests/test_models.py`

```python
import pytest
from pipeworks.ui.models import GenerationParams, SegmentConfig, UIState

def test_generation_params_validation():
    """Test parameter validation catches invalid values."""
    params = GenerationParams(
        prompt="test", width=1024, height=1024,
        num_steps=9, batch_size=1, runs=1,
        seed=42, use_random_seed=False
    )
    params.validate()  # Should not raise

    # Test invalid batch size
    params.batch_size = 150
    with pytest.raises(ValueError, match="Batch size must be 1-100"):
        params.validate()

    # Test invalid dimensions
    params.batch_size = 1
    params.width = 1023  # Not multiple of 64
    with pytest.raises(ValueError, match="Width must be multiple of 64"):
        params.validate()

def test_segment_config_is_configured():
    """Test segment configuration detection."""
    segment = SegmentConfig(file="test.txt")
    assert segment.is_configured()

    segment = SegmentConfig(file="(None)")
    assert not segment.is_configured()

    segment = SegmentConfig(file="📁 folder")
    assert not segment.is_configured()
```

**File:** `tests/test_validation.py`

```python
import pytest
from pathlib import Path
from pipeworks.ui.validation import validate_segment_path, ValidationError

def test_validate_segment_path_security(tmp_path):
    """Test path traversal prevention."""
    base_dir = tmp_path / "inputs"
    base_dir.mkdir()

    # Valid path
    test_file = base_dir / "test.txt"
    test_file.write_text("content")
    result = validate_segment_path("", "test.txt", base_dir)
    assert result == test_file

    # Path traversal attempt
    with pytest.raises(ValidationError, match="Invalid path"):
        validate_segment_path("..", "etc/passwd", base_dir)

def test_validate_segment_path_existence(tmp_path):
    """Test file existence validation."""
    base_dir = tmp_path / "inputs"
    base_dir.mkdir()

    with pytest.raises(ValidationError, match="File not found"):
        validate_segment_path("", "nonexistent.txt", base_dir)
```

**File:** `tests/conftest.py`

```python
import pytest
from pathlib import Path
from pipeworks.core.config import PipeworksConfig

@pytest.fixture
def test_config(tmp_path):
    """Create test configuration."""
    return PipeworksConfig(
        model_id="stabilityai/sdxl-turbo",
        models_dir=str(tmp_path / "models"),
        outputs_dir=str(tmp_path / "outputs"),
        inputs_dir=str(tmp_path / "inputs"),
    )

@pytest.fixture
def test_inputs_dir(tmp_path):
    """Create test inputs directory with sample files."""
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    # Create test file
    test_file = inputs_dir / "test.txt"
    test_file.write_text("line 1\nline 2\nline 3\n")

    # Create subfolder
    subfolder = inputs_dir / "subfolder"
    subfolder.mkdir()
    (subfolder / "nested.txt").write_text("nested content\n")

    return inputs_dir
```

### 5.3 GitHub Actions CI (Optional but Recommended)

**File:** `.github/workflows/test.yml`

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src/pipeworks --cov-report=term
      - name: Type check
        run: |
          mypy src/pipeworks
      - name: Lint
        run: |
          ruff check src/
```

**Benefits:**
- Regression detection
- Confidence in refactoring
- Documentation through tests
- CI ensures code quality

**Effort:** 2 days

---

## Phase 6: Refactor app.py (Priority: CRITICAL)

### 6.1 Updated Structure

**Before:**
- 1010 lines
- 8 global variables
- 37-parameter functions
- 400+ lines of duplicated code

**After:**
```
src/pipeworks/ui/
├── app.py              # Main UI (reduced to ~400 lines)
├── models.py           # Dataclasses (new, ~80 lines)
├── components.py       # Reusable UI components (new, ~150 lines)
├── validation.py       # Input validation (new, ~100 lines)
├── handlers.py         # Event handlers (new, ~200 lines)
└── state.py            # State management (new, ~100 lines)
```

### 6.2 New app.py Signature Examples

**Old:**
```python
def generate_image(
    prompt: str, width: int, height: int, num_steps: int,
    batch_size: int, runs: int, seed: int, use_random_seed: bool,
    start_text: str, start_path: str, start_file: str, start_mode: str,
    start_line: int, start_range_end: int, start_count: int, start_dynamic: bool,
    middle_text: str, middle_path: str, middle_file: str, middle_mode: str,
    middle_line: int, middle_range_end: int, middle_count: int, middle_dynamic: bool,
    end_text: str, end_path: str, end_file: str, end_mode: str,
    end_line: int, end_range_end: int, end_count: int, end_dynamic: bool,
) -> tuple[List[str], str, str]:
```

**New:**
```python
def generate_image(
    params: GenerationParams,
    start_segment: SegmentConfig,
    middle_segment: SegmentConfig,
    end_segment: SegmentConfig,
    state: UIState
) -> tuple[List[str], str, str, UIState]:
```

**Reduction:** 37 parameters → 5 clean parameters

**Effort:** 2 days

---

## Phase 7: Integration & Testing (Priority: CRITICAL)

### 7.1 Manual Testing Checklist

- [ ] Single image generation works
- [ ] Batch generation works (batch_size > 1)
- [ ] Multiple runs work (runs > 1)
- [ ] Dynamic prompts rebuild per image
- [ ] File browser navigation works
- [ ] Segment modes work (Random Line, Specific Line, etc.)
- [ ] Tokenizer analyzer updates correctly
- [ ] Aspect ratio presets work
- [ ] Plugin enable/disable works
- [ ] Metadata plugin saves files correctly
- [ ] Error messages display correctly
- [ ] Invalid inputs show validation errors
- [ ] Concurrent users don't interfere (test with 2 browser tabs)

### 7.2 Automated Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/pipeworks --cov-report=html

# Type checking
mypy src/pipeworks

# Linting
ruff check src/
black --check src/
```

**Effort:** 1 day

---

## Implementation Timeline

| Phase | Priority | Effort | Dependencies |
|-------|----------|--------|--------------|
| Phase 1: Dataclasses | CRITICAL | 1 day | None |
| Phase 2: State Management | CRITICAL | 2 days | Phase 1 |
| Phase 3: Components | HIGH | 1 day | Phase 1 |
| Phase 4: Error Handling | HIGH | 1 day | Phase 1 |
| Phase 5: Testing | HIGH | 2 days | Phases 1-4 |
| Phase 6: Refactor app.py | CRITICAL | 2 days | Phases 1-4 |
| Phase 7: Integration | CRITICAL | 1 day | All phases |
| **Total** | | **10 days** | |

**Realistic Timeline:** 10-14 days including buffer for unexpected issues

---

## Rollback Plan

If issues arise during refactoring:

1. **Branch Protection:** All changes on `refactor/fix-critical-issues` branch
2. **Main Branch:** Remains stable and deployable
3. **Incremental Merging:** Merge phases individually after testing
4. **Feature Flags:** Can add config flag to toggle between old/new implementation

```python
# Example feature flag
if config.use_legacy_ui:
    from pipeworks.ui.app_legacy import create_ui
else:
    from pipeworks.ui.app import create_ui
```

---

## Success Metrics

### Code Quality
- ✅ Zero `global` keywords in app.py
- ✅ No functions with >10 parameters
- ✅ <5 lines of duplicated code (down from 400+)
- ✅ >80% test coverage for new code
- ✅ Type hints on all public functions

### Functionality
- ✅ All existing features work
- ✅ Error messages are user-friendly
- ✅ No regression in image quality
- ✅ Performance is equal or better

### Maintainability
- ✅ New developer can understand code in <1 hour
- ✅ Adding new segment takes <30 minutes
- ✅ Bug fixes apply once, not three times

---

## Post-Refactoring: Future Improvements

After critical issues are fixed, consider:

1. **Extract Constants:** Create `ui/constants.py` for magic numbers
2. **Event Manager:** Create `UIEventManager` class to organize callbacks
3. **Graceful Degradation:** Add fallback behavior for model loading failures
4. **Rate Limiting:** Add request rate limiting for production deployment
5. **Documentation:** Add architecture diagrams and API docs

---

## Notes & Considerations

### Breaking Changes
- None! The refactoring is internal only
- UI looks and behaves identically
- Configuration remains unchanged
- No impact on users

### Performance
- Session state may use slightly more memory per user
- Offset by removing shared global state
- Model loading unchanged (still pre-loaded)
- No impact on generation speed

### Compatibility
- Python 3.10+ required (uses dataclasses with slots)
- Gradio 4.x required (uses gr.State)
- All existing dependencies remain

---

## Questions for Review

1. Should we implement all phases at once, or merge incrementally?
2. Do we need CI/CD setup, or manual testing is sufficient?
3. Should we keep the old implementation as legacy fallback?
4. Any specific edge cases to test?

---

**Status:** Ready for implementation
**Next Step:** Begin Phase 1 - Create dataclasses in `src/pipeworks/ui/models.py`
