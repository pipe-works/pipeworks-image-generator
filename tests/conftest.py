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
    ``sample_prompt_libraries`` — Minimal split prompt-library data.
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
    """Minimal models.json content with representative model definitions.

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
                "hf_url": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
                "max_prompt_tokens": 512,
                "default_steps": 4,
                "min_steps": 1,
                "max_steps": 8,
                "default_guidance": 0.0,
                "min_guidance": 0.0,
                "max_guidance": 3.0,
                "guidance_step": 0.1,
                "supports_negative_prompt": False,
                "aspect_ratios": [
                    {"id": "1:1", "label": "Square 1:1", "width": 1024, "height": 1024},
                    {"id": "16:9", "label": "Wide 16:9", "width": 1280, "height": 720},
                ],
                "default_aspect": "1:1",
            },
            {
                "id": "flux-2-klein-4b",
                "hf_id": "black-forest-labs/FLUX.2-klein-4B",
                "label": "FLUX.2-klein-4B",
                "description": "FLUX klein model for higher quality image generation.",
                "hf_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
                "max_prompt_tokens": 512,
                "default_steps": 28,
                "min_steps": 10,
                "max_steps": 50,
                "default_guidance": 0.0,
                "min_guidance": 0.0,
                "max_guidance": 0.0,
                "guidance_step": 0.1,
                "supports_negative_prompt": False,
                "aspect_ratios": [
                    {"id": "1:1", "label": "Square 1:1", "width": 1024, "height": 1024},
                    {"id": "16:9", "label": "Wide 16:9", "width": 1344, "height": 768},
                ],
                "default_aspect": "1:1",
            },
            {
                "id": "sd-v1-5",
                "hf_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "label": "Stable Diffusion v1.5",
                "description": "Classic Stable Diffusion v1.5 model.",
                "hf_url": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
                "max_prompt_tokens": 77,
                "default_steps": 20,
                "min_steps": 5,
                "max_steps": 50,
                "default_guidance": 7.5,
                "min_guidance": 1.0,
                "max_guidance": 20.0,
                "guidance_step": 0.5,
                "supports_negative_prompt": True,
                "schedulers": [
                    {"id": "pndm", "label": "PNDM (Default)"},
                    {"id": "dpmpp-2m-karras", "label": "DPM++ 2M Karras"},
                ],
                "default_scheduler": "pndm",
                "aspect_ratios": [
                    {"id": "1:1", "label": "Square 1:1", "width": 512, "height": 512},
                    {"id": "16:9", "label": "Wide 16:9", "width": 912, "height": 512},
                ],
                "default_aspect": "1:1",
            },
            {
                "id": "sdxl-1-0",
                "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "label": "SD-XL 1.0",
                "description": "Stable Diffusion XL base model.",
                "hf_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
                "max_prompt_tokens": 77,
                "default_steps": 30,
                "min_steps": 10,
                "max_steps": 60,
                "default_guidance": 7.5,
                "min_guidance": 1.0,
                "max_guidance": 20.0,
                "guidance_step": 0.5,
                "supports_negative_prompt": True,
                "aspect_ratios": [
                    {"id": "1:1", "label": "Square 1:1", "width": 1024, "height": 1024},
                    {"id": "16:9", "label": "Wide 16:9", "width": 1344, "height": 768},
                ],
                "default_aspect": "1:1",
            },
        ]
    }


@pytest.fixture
def sample_prompt_libraries() -> dict:
    """Minimal split prompt-library content with one prompt in each file."""
    return {
        "prepend": {
            "prompts": [
                {
                    "id": "oil-painting",
                    "label": "Oil Painting",
                    "value": "In a classical oil painting style.",
                },
            ]
        },
        "main": {
            "prompts": [
                {
                    "id": "goblin-workshop",
                    "label": "Goblin Workshop",
                    "value": "A goblin repairing a clockwork automaton in a dimly lit workshop.",
                },
            ]
        },
        "append": {
            "prompts": [
                {
                    "id": "high-detail",
                    "label": "High Detail",
                    "value": "Award-winning quality, 8K resolution, ultra-detailed.",
                },
            ]
        },
    }


@pytest.fixture
def tmp_data_dir(
    test_config: PipeworksConfig,
    sample_models_json: dict,
    sample_prompt_libraries: dict,
) -> Path:
    """Write sample models.json and split prompt files to the test data directory.

    Args:
        test_config: Configuration with temp paths.
        sample_models_json: Sample model definitions.
        sample_prompt_libraries: Sample split prompt-library files.

    Returns:
        Path to the data directory containing the test JSON files.
    """
    data_dir = test_config.data_dir
    with open(data_dir / "models.json", "w") as f:
        json.dump(sample_models_json, f)
    with open(data_dir / "prepend.json", "w") as f:
        json.dump(sample_prompt_libraries["prepend"], f)
    with open(data_dir / "main.json", "w") as f:
        json.dump(sample_prompt_libraries["main"], f)
    with open(data_dir / "append.json", "w") as f:
        json.dump(sample_prompt_libraries["append"], f)
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


@pytest.fixture
def mock_prompt_token_counter() -> MagicMock:
    """Create a mock prompt token counter for compile preview responses."""
    counter = MagicMock()

    def _count_prompt_parts(
        *,
        hf_id: str | None,
        prepend_text: str,
        main_text: str,
        append_text: str,
        compiled_prompt: str,
    ) -> dict:
        def _count_words(text: str) -> int:
            return len(text.split()) if text.strip() else 0

        return {
            "prepend": _count_words(prepend_text),
            "main": _count_words(main_text),
            "append": _count_words(append_text),
            "total": _count_words(compiled_prompt),
            "method": "tokenizer" if hf_id else "heuristic",
        }

    counter.count_prompt_parts.side_effect = _count_prompt_parts
    return counter


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
            "prepend_mode": "template",
            "append_mode": "template",
            "manual_prepend": None,
            "manual_append": None,
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


@pytest.fixture
def sample_gallery_mixed_models(tmp_gallery_dir: Path, test_config: PipeworksConfig) -> list[dict]:
    """Create gallery entries spanning multiple model IDs.

    This fixture is used for tests that need to verify model-filtered counts and
    pagination after reconciliation against the gallery directory.

    Args:
        tmp_gallery_dir: Path to the gallery directory.
        test_config: Configuration with temp paths.

    Returns:
        Mixed-model gallery entry dictionaries in persisted order.
    """
    import time
    import uuid

    model_ids = [
        ("z-image-turbo", "Z-Image Turbo"),
        ("sdxl-base", "SDXL Base"),
        ("z-image-turbo", "Z-Image Turbo"),
        ("sdxl-base", "SDXL Base"),
        ("z-image-turbo", "Z-Image Turbo"),
    ]

    entries: list[dict] = []

    for index, (model_id, model_label) in enumerate(model_ids):
        img_id = str(uuid.uuid4())
        filename = f"{img_id}.png"
        filepath = tmp_gallery_dir / filename

        _make_test_image().save(filepath, format="PNG")

        entries.append(
            {
                "id": img_id,
                "filename": filename,
                "url": f"/static/gallery/{filename}",
                "model_id": model_id,
                "model_label": model_label,
                "compiled_prompt": f"Prompt {index}",
                "prepend_prompt_id": "none",
                "prompt_mode": "manual",
                "manual_prompt": f"Prompt {index}",
                "automated_prompt_id": None,
                "append_prompt_id": "none",
                "prepend_mode": "template",
                "append_mode": "template",
                "manual_prepend": None,
                "manual_append": None,
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
                "seed": 100 + index,
                "negative_prompt": None,
                "is_favourite": index % 2 == 0,
                "created_at": time.time() - (index * 60),
                "batch_index": 0,
                "batch_size": 1,
                "batch_seed": 100 + index,
            }
        )

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
    mock_prompt_token_counter: MagicMock,
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
        app.state.prompt_token_counter = mock_prompt_token_counter

        client = TestClient(app)
        yield client
