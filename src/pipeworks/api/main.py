"""Pipe-Works Image Generator — FastAPI Application.

This module is the single entry point for the web application.  It defines
the FastAPI ``app`` instance, all REST API routes, and the ``main()`` CLI
function that launches the uvicorn server.

Architecture
------------
The application follows a stateless REST pattern:

- **Configuration** is loaded from JSON files on disk (``models.json``,
  ``prompts.json``) and served to the frontend via ``GET /api/config``.
- **Image generation** is performed by :class:`~pipeworks.core.model_manager.ModelManager`,
  which manages the diffusers pipeline lifecycle.
- **Gallery persistence** uses a single ``gallery.json`` file — no database
  required.
- **Static assets** (CSS, JS, fonts, generated images) are served by
  FastAPI's ``StaticFiles`` middleware.
- **The HTML page** is served as a raw ``HTMLResponse`` — no template
  engine is needed because all dynamic data is fetched via ``/api/config``
  on page load.

Endpoints
---------
========  ============================  ====================================
Method    Path                          Purpose
========  ============================  ====================================
GET       ``/``                         Serve the main HTML page
GET       ``/api/config``               Models, prompts, aspect ratios
POST      ``/api/generate``             Generate an image batch
POST      ``/api/prompt/compile``       Preview the compiled prompt
GET       ``/api/gallery``              Paginated gallery listing
GET       ``/api/gallery/{id}``         Single gallery entry
GET       ``/api/gallery/{id}/prompt``  Compiled prompt metadata
POST      ``/api/gallery/favourite``    Toggle favourite status
DELETE    ``/api/gallery/{id}``         Delete image and gallery entry
GET       ``/api/stats``                Gallery statistics
========  ============================  ====================================

Usage
-----
CLI (installed entry point)::

    pipeworks

Direct invocation::

    python -m pipeworks.api.main
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pipeworks import __version__
from pipeworks.api.models import FavouriteRequest, GenerateRequest
from pipeworks.api.prompt_builder import build_prompt
from pipeworks.core.config import config
from pipeworks.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve paths from the global configuration instance.
# ---------------------------------------------------------------------------
STATIC_DIR: Path = config.static_dir
DATA_DIR: Path = config.data_dir
GALLERY_DIR: Path = config.gallery_dir
TEMPLATES_DIR: Path = config.templates_dir
GALLERY_DB: Path = DATA_DIR / "gallery.json"

# Ensure the gallery directory exists at import time.
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Application lifecycle — model manager setup and teardown.
#
# Uses the modern lifespan context manager pattern (FastAPI ≥0.109) instead
# of the deprecated ``on_event("startup")`` / ``on_event("shutdown")``
# decorators.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    On startup:
        Initialises a :class:`ModelManager` and stores it on ``app.state``.
        No model is loaded at this point — loading happens lazily on the
        first ``POST /api/generate`` call.

    On shutdown:
        Unloads the model and frees GPU memory, ensuring CUDA resources are
        returned to the OS when the server stops.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the application for the duration of its lifetime.
    """
    # --- Startup -----------------------------------------------------------
    app.state.model_manager = ModelManager(config)
    logger.info("ModelManager initialised (no model loaded yet).")

    yield  # Application runs here.

    # --- Shutdown ----------------------------------------------------------
    app.state.model_manager.unload()
    logger.info("ModelManager unloaded on shutdown.")


# ---------------------------------------------------------------------------
# FastAPI application instance.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pipe-Works Image Generator",
    description="Image generation API with multi-model diffusion pipeline support.",
    version=__version__,
    lifespan=lifespan,
)

# Allow cross-origin requests so the frontend can be served from a different
# port during development.  In production, restrict ``allow_origins`` to the
# actual deployment domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory so that CSS, JS, fonts, and gallery images are
# served directly by FastAPI at ``/static/...``.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# JSON persistence helpers.
# ---------------------------------------------------------------------------


def _load_json(path: Path, default):
    """Load a JSON file from disk, returning *default* on any failure.

    This is intentionally forgiving: if the file does not exist, is empty,
    or contains invalid JSON, the caller gets the default value rather than
    an exception.  This makes the gallery self-bootstrapping — the first
    ``POST /api/generate`` call creates ``gallery.json`` automatically.

    Args:
        path: Absolute path to the JSON file.
        default: Value to return if the file cannot be read or parsed.

    Returns:
        The parsed JSON content, or *default*.
    """
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _save_json(path: Path, data) -> None:
    """Persist a Python object to a JSON file.

    The file is written with 2-space indentation for readability.

    Args:
        path: Absolute path to the JSON file.
        data: JSON-serialisable Python object (typically a list or dict).
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Prompt resolution helper.
# ---------------------------------------------------------------------------


def _resolve_prompt_parts(
    req: GenerateRequest,
    prompts: dict,
    *,
    strict: bool = False,
) -> tuple[str, str, str]:
    """Resolve prepend, main scene, and append values from a request.

    Handles both template (preset lookup) and manual (free-text) modes for
    the prepend and append sections, and manual/automated modes for the
    main scene.

    Args:
        req: The validated generation request.
        prompts: The loaded ``prompts.json`` data.
        strict: If ``True``, raise :class:`HTTPException` on missing or
            invalid prompt values (used by ``/api/generate``).  If ``False``,
            silently fall back to empty strings (used by ``/api/prompt/compile``).

    Returns:
        Tuple of ``(prepend_value, main_scene, append_value)``.

    Raises:
        HTTPException: (only when ``strict=True``) 400 for missing manual
            prompt, unknown automated prompt, or invalid prompt mode.
    """
    # --- Prepend resolution ------------------------------------------------
    prepend_value = ""
    if req.prepend_mode == "manual":
        prepend_value = (req.manual_prepend or "").strip()
    else:
        if req.prepend_prompt_id and req.prepend_prompt_id != "none":
            p = next(
                (x for x in prompts.get("prepend_prompts", []) if x["id"] == req.prepend_prompt_id),
                None,
            )
            if p:
                prepend_value = p["value"]

    # --- Main scene resolution ---------------------------------------------
    main_scene = ""
    if req.prompt_mode == "manual":
        if strict and (not req.manual_prompt or not req.manual_prompt.strip()):
            raise HTTPException(
                status_code=400,
                detail="manual_prompt is required in manual mode",
            )
        main_scene = (req.manual_prompt or "").strip()
    elif req.prompt_mode == "automated":
        if strict and not req.automated_prompt_id:
            raise HTTPException(
                status_code=400,
                detail="automated_prompt_id is required in automated mode",
            )
        if req.automated_prompt_id:
            ap = next(
                (
                    x
                    for x in prompts.get("automated_prompts", [])
                    if x["id"] == req.automated_prompt_id
                ),
                None,
            )
            if ap:
                main_scene = ap["value"]
            elif strict:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown automated prompt: {req.automated_prompt_id}",
                )
    elif strict:
        raise HTTPException(
            status_code=400,
            detail="prompt_mode must be 'manual' or 'automated'",
        )

    # --- Append resolution -------------------------------------------------
    append_value = ""
    if req.append_mode == "manual":
        append_value = (req.manual_append or "").strip()
    else:
        if req.append_prompt_id and req.append_prompt_id != "none":
            a = next(
                (x for x in prompts.get("append_prompts", []) if x["id"] == req.append_prompt_id),
                None,
            )
            if a:
                append_value = a["value"]

    return prepend_value, main_scene, append_value


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the main application HTML page.

    Reads ``templates/index.html`` and returns it directly.  All dynamic
    data (models, prompts, configuration) is fetched by the frontend
    JavaScript via ``GET /api/config`` on page load — no server-side
    template rendering is needed.

    Returns:
        The HTML content of the application page.

    Raises:
        HTTPException: 404 if ``index.html`` is not found.
    """
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/api/config")
async def get_config() -> dict:
    """Return the full application configuration for the frontend.

    The response includes:

    - ``version`` — API version string.
    - ``models`` — list of available model definitions (from ``models.json``),
      each including aspect ratios and inference parameter ranges.
    - ``prepend_prompts`` — style prefix presets.
    - ``automated_prompts`` — scene description presets.
    - ``append_prompts`` — post-processing modifier presets.

    Returns:
        Dictionary with keys ``version``, ``models``, ``prepend_prompts``,
        ``automated_prompts``, and ``append_prompts``.
    """
    models = _load_json(DATA_DIR / "models.json", {"models": []})
    prompts = _load_json(DATA_DIR / "prompts.json", {})
    return {
        "version": __version__,
        "models": models.get("models", []),
        "prepend_prompts": prompts.get("prepend_prompts", []),
        "automated_prompts": prompts.get("automated_prompts", []),
        "append_prompts": prompts.get("append_prompts", []),
    }


@app.post("/api/generate")
async def generate_images(req: GenerateRequest) -> dict:
    """Generate a batch of images using the selected diffusion model.

    This endpoint:

    1. Validates the batch size (1–16).
    2. Resolves the model configuration from ``models.json``.
    3. Compiles the three-part prompt (prepend + scene + append).
    4. Loads the model if not already loaded (or switches models).
    5. Generates ``batch_size`` images with incrementing seeds.
    6. Saves each image to the gallery and persists metadata.

    Args:
        req: Validated :class:`GenerateRequest` payload.

    Returns:
        Dictionary with keys ``success``, ``batch_seed``,
        ``compiled_prompt``, and ``images`` (list of gallery entries).

    Raises:
        HTTPException: 400 for invalid batch size, unknown model, missing
            prompt, or unknown prompt mode.
    """
    # --- Validate batch size -----------------------------------------------
    if req.batch_size < 1 or req.batch_size > 16:
        raise HTTPException(
            status_code=400,
            detail="batch_size must be between 1 and 16",
        )

    # --- Load configuration data -------------------------------------------
    prompts = _load_json(DATA_DIR / "prompts.json", {})
    models_data = _load_json(DATA_DIR / "models.json", {"models": []})

    # --- Resolve model configuration ---------------------------------------
    # Find the model definition that matches the requested model_id.
    model_cfg = next(
        (m for m in models_data.get("models", []) if m["id"] == req.model_id),
        None,
    )
    if not model_cfg:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {req.model_id}",
        )

    # --- Resolve prompt parts and compile ------------------------------------
    prepend_value, main_scene, append_value = _resolve_prompt_parts(req, prompts, strict=True)
    compiled_prompt = build_prompt(
        prepend_value,
        main_scene,
        append_value,
        prepend_mode=req.prepend_mode,
        append_mode=req.append_mode,
    )

    # --- Resolve seed ------------------------------------------------------
    # If the client did not supply a seed, pick a random one.  Each image in
    # the batch gets seed = base_seed + batch_index for reproducibility.
    base_seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

    # --- Ensure model is loaded --------------------------------------------
    model_mgr: ModelManager = app.state.model_manager
    hf_id = model_cfg["hf_id"]

    # Switch models if the currently loaded model differs from the request.
    if model_mgr.current_model_id != hf_id:
        model_mgr.load_model(hf_id)

    # --- Generate batch ----------------------------------------------------
    gallery = _load_json(GALLERY_DB, [])
    generated: list[dict] = []

    for i in range(req.batch_size):
        # Each image in the batch gets a unique, incrementing seed.
        img_seed = base_seed + i
        img_id = str(uuid.uuid4())
        filename = f"{img_id}.png"
        filepath = GALLERY_DIR / filename

        # Generate the image using the real diffusion pipeline.
        image = model_mgr.generate(
            prompt=compiled_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance,
            seed=img_seed,
            negative_prompt=req.negative_prompt,
        )

        # Save the PIL image to disk as a PNG file.
        image.save(filepath, format="PNG")

        # Build the gallery metadata entry.
        entry = {
            "id": img_id,
            "filename": filename,
            "url": f"/static/gallery/{filename}",
            "model_id": req.model_id,
            "model_label": model_cfg["label"],
            "compiled_prompt": compiled_prompt,
            "prepend_prompt_id": req.prepend_prompt_id,
            "prompt_mode": req.prompt_mode,
            "manual_prompt": req.manual_prompt,
            "automated_prompt_id": req.automated_prompt_id,
            "append_prompt_id": req.append_prompt_id,
            "aspect_ratio_id": req.aspect_ratio_id,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance": req.guidance,
            "seed": img_seed,
            "negative_prompt": req.negative_prompt,
            "is_favourite": False,
            "created_at": time.time(),
            "batch_index": i,
            "batch_size": req.batch_size,
            "batch_seed": base_seed,
        }

        # Insert newest first so the gallery is in reverse-chronological order.
        gallery.insert(0, entry)
        generated.append(entry)

    # Persist the updated gallery to disk.
    _save_json(GALLERY_DB, gallery)

    return {
        "success": True,
        "batch_seed": base_seed,
        "compiled_prompt": compiled_prompt,
        "images": generated,
    }


@app.get("/api/gallery")
async def get_gallery(
    page: int = 1,
    per_page: int = 20,
    favourites_only: bool = False,
    model_id: str | None = None,
) -> dict:
    """Return a paginated listing of gallery images.

    Supports filtering by favourite status and model ID.

    Args:
        page: Page number (1-indexed).
        per_page: Number of images per page.
        favourites_only: If ``True``, return only favourited images.
        model_id: If provided, return only images from this model.

    Returns:
        Dictionary with keys ``total``, ``page``, ``per_page``, ``pages``,
        and ``images``.
    """
    gallery = _load_json(GALLERY_DB, [])

    # Apply filters.
    if favourites_only:
        gallery = [g for g in gallery if g.get("is_favourite")]
    if model_id:
        gallery = [g for g in gallery if g.get("model_id") == model_id]

    # Calculate pagination.
    total = len(gallery)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = gallery[start:end]

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if total > 0 else 1,
        "images": page_items,
    }


@app.get("/api/gallery/{image_id}")
async def get_image(image_id: str) -> dict:
    """Return a single gallery entry by UUID.

    Args:
        image_id: UUID of the gallery image.

    Returns:
        The gallery entry dictionary.

    Raises:
        HTTPException: 404 if the image is not found.
    """
    gallery: list[dict] = _load_json(GALLERY_DB, [])
    entry: dict | None = next((g for g in gallery if g["id"] == image_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")
    return entry


@app.post("/api/gallery/favourite")
async def toggle_favourite(req: FavouriteRequest) -> dict:
    """Toggle the favourite status of a gallery image.

    Args:
        req: Validated :class:`FavouriteRequest` payload.

    Returns:
        Dictionary with ``success``, ``id``, and ``is_favourite``.

    Raises:
        HTTPException: 404 if the image is not found.
    """
    gallery = _load_json(GALLERY_DB, [])
    entry = next((g for g in gallery if g["id"] == req.image_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")

    # Update the favourite flag and persist to disk.
    entry["is_favourite"] = req.is_favourite
    _save_json(GALLERY_DB, gallery)

    return {"success": True, "id": req.image_id, "is_favourite": req.is_favourite}


@app.delete("/api/gallery/{image_id}")
async def delete_image(image_id: str) -> dict:
    """Delete an image from the gallery.

    Removes both the PNG file from disk and the metadata entry from
    ``gallery.json``.

    Args:
        image_id: UUID of the gallery image to delete.

    Returns:
        Dictionary with ``success`` and ``deleted`` keys.

    Raises:
        HTTPException: 404 if the image is not found.
    """
    gallery = _load_json(GALLERY_DB, [])
    entry = next((g for g in gallery if g["id"] == image_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")

    # Remove the PNG file from disk.
    filepath = GALLERY_DIR / entry["filename"]
    if filepath.exists():
        filepath.unlink()

    # Remove the entry from the gallery list and persist.
    gallery = [g for g in gallery if g["id"] != image_id]
    _save_json(GALLERY_DB, gallery)

    return {"success": True, "deleted": image_id}


@app.get("/api/gallery/{image_id}/prompt")
async def get_image_prompt(image_id: str) -> dict:
    """Return the compiled prompt and generation metadata for an image.

    Args:
        image_id: UUID of the gallery image.

    Returns:
        Dictionary with ``id``, ``compiled_prompt``, ``model_id``,
        ``seed``, ``steps``, ``guidance``, ``width``, and ``height``.

    Raises:
        HTTPException: 404 if the image is not found.
    """
    gallery = _load_json(GALLERY_DB, [])
    entry = next((g for g in gallery if g["id"] == image_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")
    return {
        "id": image_id,
        "compiled_prompt": entry.get("compiled_prompt", ""),
        "model_id": entry.get("model_id"),
        "seed": entry.get("seed"),
        "steps": entry.get("steps"),
        "guidance": entry.get("guidance"),
        "width": entry.get("width"),
        "height": entry.get("height"),
    }


@app.post("/api/prompt/compile")
async def compile_prompt(req: GenerateRequest) -> dict:
    """Preview the compiled prompt without generating an image.

    Resolves the prepend, main scene, and append parts from the request
    and returns the fully compiled prompt string.

    Args:
        req: Validated :class:`GenerateRequest` payload (only prompt-related
            fields are used).

    Returns:
        Dictionary with a single ``compiled_prompt`` key.
    """
    prompts = _load_json(DATA_DIR / "prompts.json", {})
    prepend_value, main_scene, append_value = _resolve_prompt_parts(req, prompts, strict=False)
    compiled = build_prompt(
        prepend_value,
        main_scene,
        append_value,
        prepend_mode=req.prepend_mode,
        append_mode=req.append_mode,
    )
    return {"compiled_prompt": compiled}


@app.get("/api/stats")
async def get_stats() -> dict:
    """Return gallery statistics.

    Returns:
        Dictionary with ``total_images``, ``total_favourites``, and
        ``model_counts`` (a mapping of model_id → count).
    """
    gallery = _load_json(GALLERY_DB, [])
    total = len(gallery)
    favourites = sum(1 for g in gallery if g.get("is_favourite"))

    # Count images per model.
    model_counts: dict[str, int] = {}
    for g in gallery:
        mid = g.get("model_id", "unknown")
        model_counts[mid] = model_counts.get(mid, 0) + 1

    return {
        "total_images": total,
        "total_favourites": favourites,
        "model_counts": model_counts,
    }


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the uvicorn ASGI server.

    Reads host and port from :data:`~pipeworks.core.config.config` (which
    loads from ``PIPEWORKS_SERVER_HOST`` and ``PIPEWORKS_SERVER_PORT``
    environment variables).  Defaults to ``0.0.0.0:7860``.

    This function is registered as the ``pipeworks`` console script in
    ``pyproject.toml``.
    """
    import uvicorn

    uvicorn.run(
        "pipeworks.api.main:app",
        host=config.server_host,
        port=config.server_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
