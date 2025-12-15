"""Shared pytest fixtures for Pipeworks tests."""

import pytest
from pathlib import Path
import tempfile
import shutil
from typing import Generator

from pipeworks.core.config import PipeworksConfig
from pipeworks.ui.models import GenerationParams, SegmentConfig, UIState


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Yields:
        Path to temporary directory

    Cleanup:
        Directory is removed after test completes
    """
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir: Path) -> PipeworksConfig:
    """Create a test configuration with temporary directories.

    Args:
        temp_dir: Temporary directory from fixture

    Returns:
        PipeworksConfig instance for testing
    """
    inputs_dir = temp_dir / "inputs"
    outputs_dir = temp_dir / "outputs"
    models_dir = temp_dir / "models"

    inputs_dir.mkdir()
    outputs_dir.mkdir()
    models_dir.mkdir()

    return PipeworksConfig(
        model_id="stabilityai/sdxl-turbo",  # Won't actually load in tests
        models_dir=str(models_dir),
        outputs_dir=str(outputs_dir),
        inputs_dir=str(inputs_dir),
        device="cpu",  # Use CPU for tests
        torch_dtype="float32",
        default_width=1024,
        default_height=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
    )


@pytest.fixture
def test_inputs_dir(temp_dir: Path) -> Path:
    """Create a test inputs directory with sample files.

    Args:
        temp_dir: Temporary directory from fixture

    Returns:
        Path to inputs directory with test files
    """
    inputs_dir = temp_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    # Create test file with lines
    test_file = inputs_dir / "test.txt"
    test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

    # Create another test file
    test_file2 = inputs_dir / "test2.txt"
    test_file2.write_text("alpha\nbeta\ngamma\ndelta\n")

    # Create subfolder with file
    subfolder = inputs_dir / "subfolder"
    subfolder.mkdir()
    nested_file = subfolder / "nested.txt"
    nested_file.write_text("nested line 1\nnested line 2\n")

    # Create deeper nesting
    deep_folder = subfolder / "deep"
    deep_folder.mkdir()
    deep_file = deep_folder / "deep.txt"
    deep_file.write_text("deep content\n")

    return inputs_dir


@pytest.fixture
def valid_generation_params() -> GenerationParams:
    """Create valid generation parameters for testing.

    Returns:
        GenerationParams with valid values
    """
    return GenerationParams(
        prompt="A test prompt",
        width=1024,
        height=1024,
        num_steps=9,
        batch_size=1,
        runs=1,
        seed=42,
        use_random_seed=False
    )


@pytest.fixture
def valid_segment_config() -> SegmentConfig:
    """Create valid segment configuration for testing.

    Returns:
        SegmentConfig with valid values
    """
    return SegmentConfig(
        text="test text",
        path="",
        file="test.txt",
        mode="Random Line",
        line=1,
        range_end=1,
        count=1,
        dynamic=False
    )


@pytest.fixture
def empty_segment_config() -> SegmentConfig:
    """Create empty segment configuration for testing.

    Returns:
        SegmentConfig with no content
    """
    return SegmentConfig()


@pytest.fixture
def ui_state() -> UIState:
    """Create empty UI state for testing.

    Returns:
        UIState instance
    """
    return UIState()


@pytest.fixture
def sample_prompts() -> list[str]:
    """Sample prompts for testing.

    Returns:
        List of test prompts
    """
    return [
        "A simple prompt",
        "A prompt with multiple words and details",
        "Short",
        "A very long prompt " * 20,  # Long prompt for testing limits
    ]
