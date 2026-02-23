"""Shared pytest fixtures for Pipe-Works Image Generator tests.

This module provides reusable fixtures for both unit and integration tests.
Fixtures are designed around the FastAPI + ModelManager architecture and use
temporary directories to avoid polluting the real file system.

Fixture Categories
------------------
Configuration
    ``test_config`` — PipeworksConfig instance with temp directories.
    ``tmp_data_dir`` — Temporary data directory with sample JSON files.
    ``tmp_gallery_dir`` — Temporary gallery directory.
    ``tmp_templates_dir`` — Temporary templates directory with index.html.

Model Manager
    ``mock_model_manager`` — ModelManager mock that returns a 64x64 red image.

Gallery
    ``sample_gallery`` — 5 pre-populated gallery entries with placeholder PNGs.

FastAPI
    ``test_client`` — FastAPI TestClient with mocked ModelManager and temp paths.

Sample Data
    ``sample_models_json`` — Minimal models.json with one model definition.
    ``sample_prompts_json`` — Minimal prompts.json with one of each prompt type.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pipeworks.core.config import PipeworksConfig

# ---------------------------------------------------------------------------
# Temporary directory fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create and yield a temporary directory, cleaned up after the test.

    Yields:
        Path to a freshly created temporary directory.
    """
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Configuration fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def test_config(temp_dir: Path) -> PipeworksConfig:
    """Create a PipeworksConfig instance with all paths pointing to temp dirs.

    This prevents tests from touching real model caches, gallery directories,
    or output folders.  The device is set to ``cpu`` and dtype to ``float32``
    so that tests can run on any machine without a GPU.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Fully initialised PipeworksConfig for testing.
    """
    models_dir = temp_dir / "models"
    outputs_dir = temp_dir / "outputs"
    static_dir = temp_dir / "static"
    data_dir = static_dir / "data"
    gallery_dir = static_dir / "gallery"
    templates_dir = temp_dir / "templates"

    # Create all necessary directories.
    for d in [models_dir, outputs_dir, data_dir, gallery_dir, templates_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return PipeworksConfig(
        models_dir=str(models_dir),
        outputs_dir=str(outputs_dir),
        static_dir=str(static_dir),
        data_dir=str(data_dir),
        gallery_dir=str(gallery_dir),
        templates_dir=str(templates_dir),
        device="cpu",
        torch_dtype="float32",
        default_width=1024,
        default_height=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        server_host="127.0.0.1",
        server_port=7860,
    )


# ---------------------------------------------------------------------------
# Sample JSON data fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_models_json() -> dict:
    """Minimal models.json content with a single model definition.

    Returns:
        Dictionary matching the models.json schema.
    """
    return {
        "models": [
            {
                "id": "z-image-turbo",
                "hf_id": "Tongyi-MAI/Z-Image-Turbo",
                "label": "Z-Image Turbo",
                "description": "Fast turbo model for quick generation.",
                "aspect_ratios": [
                    {"id": "1:1", "label": "Square 1:1", "width": 1024, "height": 1024},
                    {"id": "16:9", "label": "Wide 16:9", "width": 1280, "height": 720},
                ],
                "defaults": {
                    "steps": 4,
                    "guidance": 0.0,
                    "aspect_ratio": "1:1",
                },
            }
        ]
    }


@pytest.fixture
def sample_prompts_json() -> dict:
    """Minimal prompts.json content with one of each prompt type.

    Returns:
        Dictionary matching the prompts.json schema.
    """
    return {
        "prepend_prompts": [
            {
                "id": "oil-painting",
                "label": "Oil Painting",
                "value": "In a classical oil painting style.",
            },
        ],
        "automated_prompts": [
            {
                "id": "goblin-workshop",
                "label": "Goblin Workshop",
                "value": "A goblin repairing a clockwork automaton in a dimly lit workshop.",
            },
        ],
        "append_prompts": [
            {
                "id": "high-detail",
                "label": "High Detail",
                "value": "Award-winning quality, 8K resolution, ultra-detailed.",
            },
        ],
    }


@pytest.fixture
def tmp_data_dir(
    test_config: PipeworksConfig,
    sample_models_json: dict,
    sample_prompts_json: dict,
) -> Path:
    """Write sample models.json and prompts.json to the test data directory.

    Args:
        test_config: Configuration with temp paths.
        sample_models_json: Sample model definitions.
        sample_prompts_json: Sample prompt presets.

    Returns:
        Path to the data directory containing both JSON files.
    """
    data_dir = test_config.data_dir
    with open(data_dir / "models.json", "w") as f:
        json.dump(sample_models_json, f)
    with open(data_dir / "prompts.json", "w") as f:
        json.dump(sample_prompts_json, f)
    return data_dir


@pytest.fixture
def tmp_gallery_dir(test_config: PipeworksConfig) -> Path:
    """Return the test gallery directory (already created by test_config).

    Args:
        test_config: Configuration with temp paths.

    Returns:
        Path to the gallery directory.
    """
    return test_config.gallery_dir


@pytest.fixture
def tmp_templates_dir(test_config: PipeworksConfig) -> Path:
    """Create a minimal index.html in the test templates directory.

    Args:
        test_config: Configuration with temp paths.

    Returns:
        Path to the templates directory containing index.html.
    """
    templates_dir = test_config.templates_dir
    (templates_dir / "index.html").write_text(
        "<html><body><h1>Pipe-Works Image Generator</h1></body></html>"
    )
    return templates_dir


# ---------------------------------------------------------------------------
# Model manager fixtures.
# ---------------------------------------------------------------------------


def _make_test_image(width: int = 64, height: int = 64) -> Image.Image:
    """Create a small solid-red PIL image for testing.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A PIL Image filled with solid red (#FF0000).
    """
    return Image.new("RGB", (width, height), color=(255, 0, 0))


@pytest.fixture
def mock_model_manager() -> MagicMock:
    """Create a mock ModelManager that returns a 64x64 red test image.

    The mock tracks all calls to ``load_model()``, ``generate()``, and
    ``unload()``.  The ``generate()`` method returns a fresh 64x64 red
    PIL image on each call.

    Returns:
        MagicMock instance configured as a ModelManager substitute.
    """
    mgr = MagicMock()
    mgr.is_loaded = False
    mgr.current_model_id = None

    # generate() returns a fresh test image each time.
    mgr.generate.side_effect = lambda **kwargs: _make_test_image(
        kwargs.get("width", 64), kwargs.get("height", 64)
    )

    return mgr


# ---------------------------------------------------------------------------
# Gallery fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_gallery(tmp_gallery_dir: Path, test_config: PipeworksConfig) -> list[dict]:
    """Create 5 pre-populated gallery entries with placeholder PNG files.

    Each entry has a unique UUID, metadata, and a corresponding 64x64 red
    PNG file in the gallery directory.  The gallery.json is written to
    the data directory.

    Args:
        tmp_gallery_dir: Path to the gallery directory.
        test_config: Configuration with temp paths.

    Returns:
        List of 5 gallery entry dictionaries (newest first).
    """
    import time
    import uuid

    entries: list[dict] = []
    for i in range(5):
        img_id = str(uuid.uuid4())
        filename = f"{img_id}.png"
        filepath = tmp_gallery_dir / filename

        # Write a small red PNG placeholder.
        _make_test_image().save(filepath, format="PNG")

        entry = {
            "id": img_id,
            "filename": filename,
            "url": f"/static/gallery/{filename}",
            "model_id": "z-image-turbo",
            "model_label": "Z-Image Turbo",
            "compiled_prompt": f"Test prompt {i}",
            "prepend_prompt_id": "none",
            "prompt_mode": "manual",
            "manual_prompt": f"Test prompt {i}",
            "automated_prompt_id": None,
            "append_prompt_id": "none",
            "aspect_ratio_id": "1:1",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 0.0,
            "seed": 42 + i,
            "negative_prompt": None,
            "is_favourite": i == 0,  # First entry is favourited.
            "created_at": time.time() - (i * 60),
            "batch_index": 0,
            "batch_size": 1,
            "batch_seed": 42 + i,
        }
        entries.append(entry)

    # Write gallery.json to the data directory.
    gallery_db = test_config.data_dir / "gallery.json"
    with open(gallery_db, "w") as f:
        json.dump(entries, f, indent=2)

    return entries


# ---------------------------------------------------------------------------
# FastAPI TestClient fixture.
# ---------------------------------------------------------------------------


@pytest.fixture
def test_client(
    test_config: PipeworksConfig,
    mock_model_manager: MagicMock,
    tmp_data_dir: Path,
    tmp_templates_dir: Path,
):
    """Create a FastAPI TestClient with mocked dependencies.

    Patches the module-level path constants and the ModelManager so that
    tests run against temporary directories without touching real files
    or loading real models.

    Args:
        test_config: Configuration with temp paths.
        mock_model_manager: Mocked ModelManager.
        tmp_data_dir: Data directory with sample JSON files.
        tmp_templates_dir: Templates directory with index.html.

    Yields:
        ``httpx``-based TestClient for the FastAPI application.
    """
    from fastapi.testclient import TestClient

    # Patch the module-level path constants in pipeworks.api.main so that
    # routes read from / write to our temporary directories.
    with (
        patch("pipeworks.api.main.STATIC_DIR", test_config.static_dir),
        patch("pipeworks.api.main.DATA_DIR", test_config.data_dir),
        patch("pipeworks.api.main.GALLERY_DIR", test_config.gallery_dir),
        patch("pipeworks.api.main.TEMPLATES_DIR", test_config.templates_dir),
        patch("pipeworks.api.main.GALLERY_DB", test_config.data_dir / "gallery.json"),
    ):
        from pipeworks.api.main import app

        # Inject the mock model manager onto app.state.
        app.state.model_manager = mock_model_manager

        client = TestClient(app)
        yield client
