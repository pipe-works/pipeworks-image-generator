"""Pipe-Works Image Generator — FastAPI Application.

This module is the single entry point for the web application.  It defines
the FastAPI ``app`` instance, all REST API routes, and the ``main()`` CLI
function that launches the uvicorn server.

Architecture
------------
The application follows a stateless REST pattern:

- **Configuration** is loaded from JSON files on disk (``models.json``,
  ``prepend.json``, ``main.json``, ``append.json``) and served to the
  frontend via ``GET /api/config``.
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
GET       ``/api/gallery/{id}/zip``     Download image + metadata zip
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

import io
import json
import logging
import random
import threading
import time
import uuid
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from pipeworks import __version__
from pipeworks.api.gallery_store import (
    filter_gallery_entries,
    get_run_entries,
    group_entries_into_runs,
    load_gallery_entries,
    paginate_gallery_entries,
    paginate_runs,
    save_gallery_entries,
)
from pipeworks.api.models import (
    BulkDeleteRequest,
    BulkZipRequest,
    CancelGenerationRequest,
    FavouriteRequest,
    GenerateRequest,
)
from pipeworks.api.prompt_builder import (
    SECTION_ORDER,
    build_prompt,
    build_structured_prompt,
    expand_prompt_placeholders,
    resolve_prompt_variants,
    resolve_structured_prompt_variants,
)
from pipeworks.core.config import config
from pipeworks.core.model_manager import ModelManager, get_model_runtime_support
from pipeworks.core.prompt_token_counter import PromptTokenCounter

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 1000
PROMPT_SECTION_ORDER = SECTION_ORDER

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
    app.state.prompt_token_counter = PromptTokenCounter(config)
    app.state.generation_cancel_events = {}
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


@app.middleware("http")
async def disable_http_cache_for_local_testing(request, call_next):
    """Optionally disable browser caching for local development.

    When ``PIPEWORKS_DISABLE_HTTP_CACHE=true`` the app emits no-cache headers
    for the HTML shell, API responses, and static assets.  This keeps browser
    refreshes honest while iterating on CSS/JS locally without changing the
    default production caching behaviour.
    """
    response = await call_next(request)

    if config.disable_http_cache and (
        request.url.path == "/"
        or request.url.path.startswith("/api/")
        or request.url.path.startswith("/static/")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response


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


def _load_prompt_list(path: Path, *, legacy_prompts: dict, legacy_key: str) -> list[dict]:
    """Load one prompt-library file with optional legacy fallback support."""
    data = _load_json(path, None)

    prompts: list[dict] = []
    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict):
        prompts = data.get("prompts", [])

    if not prompts:
        prompts = legacy_prompts.get(legacy_key, [])

    normalized: list[dict] = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        item = dict(prompt)
        item.setdefault("source_section", path.stem)
        normalized.append(item)

    return normalized


def _merge_prompt_lists(*prompt_lists: list[dict]) -> list[dict]:
    """Merge prompt lists while preserving order and de-duplicating by id."""
    merged: list[dict] = []
    seen_ids: set[str] = set()

    for prompt_list in prompt_lists:
        for prompt in prompt_list:
            prompt_id = prompt.get("id")
            if not prompt_id:
                continue
            if prompt_id in seen_ids:
                logger.warning(
                    "Duplicate prompt id '%s' detected; keeping first occurrence.", prompt_id
                )
                continue
            seen_ids.add(prompt_id)
            merged.append(prompt)

    return merged


def _annotate_models_with_runtime_support(models: list[dict]) -> list[dict]:
    """Attach runtime availability metadata to model config entries."""
    annotated: list[dict] = []
    for model in models:
        item = dict(model)
        is_available, reason = get_model_runtime_support(item.get("hf_id", ""))
        item["is_available"] = is_available
        item["unavailable_reason"] = reason
        annotated.append(item)
    return annotated


def _load_prompt_catalog() -> dict:
    """Load prompt libraries from split files and expose a merged selector catalog."""
    legacy_prompts = _load_json(DATA_DIR / "prompts.json", {})

    prepend_library = _load_prompt_list(
        DATA_DIR / "prepend.json",
        legacy_prompts=legacy_prompts,
        legacy_key="prepend_prompts",
    )
    main_library = _load_prompt_list(
        DATA_DIR / "main.json",
        legacy_prompts=legacy_prompts,
        legacy_key="automated_prompts",
    )
    append_library = _load_prompt_list(
        DATA_DIR / "append.json",
        legacy_prompts=legacy_prompts,
        legacy_key="append_prompts",
    )

    merged_prompts = _merge_prompt_lists(prepend_library, main_library, append_library)

    return {
        "prepend_library": prepend_library,
        "main_library": main_library,
        "append_library": append_library,
        "all_prompts": merged_prompts,
        # Each selector gets the full merged catalog so prompts can be reused
        # across prepend, main, and append in any order.
        "prepend_prompts": merged_prompts,
        "automated_prompts": merged_prompts,
        "append_prompts": merged_prompts,
    }


def _resolve_policy_root() -> Path | None:
    """Resolve the world-policy root used for section dropdown snippets."""
    candidates = [
        Path.cwd() / "data/worlds/pipeworks_web/policies",
        Path(__file__).resolve().parents[3] / "data/worlds/pipeworks_web/policies",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _format_policy_label(path: Path) -> str:
    """Convert a policy file path into a short human-readable label."""
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    if not stem:
        return path.name
    return " ".join(part.capitalize() for part in stem.split())


def _extract_yaml_prompt_text(raw_yaml: str) -> str:
    """Extract a prompt string from a YAML `text` field.

    Supports the common forms used by policy blocks:
    - multiline block scalars: `text: |`
    - inline scalar: `text: some text`
    """
    lines = raw_yaml.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if not stripped.startswith("text:"):
            continue

        remainder = stripped[len("text:") :].strip()
        if remainder and remainder[0] not in {"|", ">"}:
            return remainder.strip("'\"").strip()

        block_lines: list[str] = []
        block_indent: int | None = None
        for block_line in lines[index + 1 :]:
            block_stripped = block_line.lstrip()
            current_indent = len(block_line) - len(block_stripped)

            if block_stripped == "":
                if block_indent is not None:
                    block_lines.append("")
                continue

            if current_indent <= indent:
                break

            if block_indent is None:
                block_indent = current_indent

            block_lines.append(block_line[block_indent:].rstrip())

        return "\n".join(block_lines).strip()

    return ""


def _load_policy_prompt_options() -> list[dict]:
    """Load prompt snippets from the policy tree.

    Each option includes:
    - ``id``: stable path-like identifier relative to policy root
    - ``label``: user-facing short label from filename
    - ``value``: file contents
    - ``group``: relative directory path for UI optgroup rendering

    Supported sources:
    - `.txt` files (full file text)
    - `.yaml`/`.yml` files that contain a prompt-bearing `text` field
    """
    policy_root = _resolve_policy_root()
    if policy_root is None:
        return []

    options: list[dict] = []
    supported_extensions = {".txt", ".yaml", ".yml"}
    for file_path in sorted(path for path in policy_root.rglob("*") if path.is_file()):
        if file_path.suffix.lower() not in supported_extensions:
            continue

        rel_path = file_path.relative_to(policy_root).as_posix()
        group_rel = file_path.parent.relative_to(policy_root).as_posix()
        group = group_rel if group_rel != "." else "policies"

        try:
            raw_value = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Unable to read policy prompt snippet '%s': %s", file_path, exc)
            continue

        if file_path.suffix.lower() == ".txt":
            value = raw_value.strip()
        else:
            value = _extract_yaml_prompt_text(raw_value)

        if not value:
            continue
        options.append(
            {
                "id": rel_path,
                "label": _format_policy_label(file_path),
                "value": value,
                "group": group,
                "path": rel_path,
            }
        )
    return options


def _load_policy_prompt_groups() -> list[str]:
    """Load all policy directory paths for dropdown group mirroring."""
    policy_root = _resolve_policy_root()
    if policy_root is None:
        return []

    groups = ["policies"]
    for dir_path in sorted(path for path in policy_root.rglob("*") if path.is_dir()):
        rel_path = dir_path.relative_to(policy_root).as_posix()
        groups.append(rel_path if rel_path != "." else "policies")
    return groups


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
        prompts: The loaded prompt catalog from ``prepend.json``,
            ``main.json``, and ``append.json``.
        strict: If ``True``, raise :class:`HTTPException` on missing or
            invalid prompt values (used by ``/api/generate``).  If ``False``,
            silently fall back to empty strings (used by ``/api/prompt/compile``).

    Returns:
        Tuple of ``(prepend_value, main_scene, append_value)``.

    Raises:
        HTTPException: (only when ``strict=True``) 400 for missing manual
            prompt, unknown automated prompt, or invalid prompt mode.
    """
    prompt_lookup = {
        prompt["id"]: prompt for prompt in prompts.get("all_prompts", []) if prompt.get("id")
    }

    # --- Prepend resolution ------------------------------------------------
    prepend_value = ""
    if req.prepend_mode == "manual":
        prepend_value = (req.manual_prepend or "").strip()
    else:
        if req.prepend_prompt_id and req.prepend_prompt_id != "none":
            p = prompt_lookup.get(req.prepend_prompt_id)
            if p:
                prepend_value = p["value"]

    # --- Main scene resolution ---------------------------------------------
    main_scene = ""
    if req.prompt_mode == "manual":
        main_scene = (req.manual_prompt or "").strip()
    elif req.prompt_mode == "automated":
        if req.automated_prompt_id and req.automated_prompt_id != "none":
            ap = prompt_lookup.get(req.automated_prompt_id)
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
            a = prompt_lookup.get(req.append_prompt_id)
            if a:
                append_value = a["value"]

    return prepend_value, main_scene, append_value


def _request_uses_section_schema(req: GenerateRequest) -> bool:
    """Return True when the request includes five-section composer fields."""
    if req.prompt_schema_version == 2:
        return True

    for section in PROMPT_SECTION_ORDER:
        if getattr(req, f"{section}_mode", None) is not None:
            return True
        if getattr(req, f"manual_{section}", None):
            return True
        if getattr(req, f"automated_{section}_prompt_id", None):
            return True
    return False


def _build_prompt_lookup(
    prompts: dict,
    policy_options: list[dict] | None = None,
) -> dict[str, dict]:
    """Build a prompt lookup map from legacy libraries and policy snippets."""
    lookup: dict[str, dict] = {}
    for prompt in prompts.get("all_prompts", []):
        prompt_id = prompt.get("id")
        if not prompt_id:
            continue
        lookup[prompt_id] = prompt

    for option in policy_options or []:
        option_id = option.get("id")
        if not option_id or option_id in lookup:
            continue
        lookup[option_id] = option

    return lookup


def _resolve_structured_prompt_sections(
    req: GenerateRequest,
    prompts: dict,
    *,
    policy_options: list[dict] | None = None,
    strict: bool = False,
) -> dict[str, str]:
    """Resolve Subject/Setting/Details/Lighting/Atmosphere values from request."""
    prompt_lookup = _build_prompt_lookup(prompts, policy_options)
    resolved: dict[str, str] = {}

    for section in PROMPT_SECTION_ORDER:
        mode = getattr(req, f"{section}_mode", None) or "manual"
        manual_value = (getattr(req, f"manual_{section}", None) or "").strip()
        prompt_id = getattr(req, f"automated_{section}_prompt_id", None)

        if manual_value:
            resolved[section] = manual_value
            continue

        if mode == "automated" and prompt_id and prompt_id != "none":
            prompt = prompt_lookup.get(prompt_id)
            if prompt:
                resolved[section] = (prompt.get("value") or "").strip()
            elif strict:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown automated {section} prompt: {prompt_id}",
                )
            else:
                resolved[section] = ""
            continue

        resolved[section] = ""

    return resolved


def _build_zip_metadata_for_entry(entry: dict) -> dict:
    """Build the structured metadata dictionary for a gallery entry's zip.

    This helper is shared by the single-image zip endpoint and the bulk run
    zip endpoint to ensure consistent metadata format across both.

    Args:
        entry: A single gallery entry dictionary.

    Returns:
        Structured metadata dictionary ready for JSON serialisation.
    """
    image_id = entry["id"]

    # Look up the model definition for the label.
    models_data = _load_json(DATA_DIR / "models.json", {"models": []})
    model_cfg = next(
        (m for m in models_data.get("models", []) if m["id"] == entry.get("model_id")),
        None,
    )

    created_at_ts = entry.get("created_at")
    created_at_iso = (
        datetime.fromtimestamp(created_at_ts, tz=UTC).isoformat() if created_at_ts else None
    )

    # Load prompt sources so we can resolve labels/text for metadata sections.
    prompts = _load_prompt_catalog()
    policy_options = _load_policy_prompt_options()
    prompt_lookup = _build_prompt_lookup(prompts, policy_options)

    prepend_mode = entry.get("prepend_mode", "template")
    append_mode = entry.get("append_mode", "template")

    # Resolve prepend section.
    prepend_id = entry.get("prepend_prompt_id")
    prepend_preset = prompt_lookup.get(prepend_id, {}) if prepend_id else {}
    if prepend_mode == "manual":
        prepend_text = entry.get("manual_prepend", "")
    else:
        prepend_text = prepend_preset.get("value", "")

    # Resolve main section.
    prompt_mode = entry.get("prompt_mode", "manual")
    automated_id = entry.get("automated_prompt_id")
    automated_preset = prompt_lookup.get(automated_id, {}) if automated_id else {}
    if prompt_mode == "manual":
        main_text = entry.get("manual_prompt", "") or ""
    else:
        main_text = automated_preset.get("value", "")

    # Resolve append section.
    append_id = entry.get("append_prompt_id")
    append_preset = prompt_lookup.get(append_id, {}) if append_id else {}
    if append_mode == "manual":
        append_text = entry.get("manual_append", "")
    else:
        append_text = append_preset.get("value", "")

    sections_metadata: dict[str, dict] = {}
    for section in PROMPT_SECTION_ORDER:
        section_mode = entry.get(f"{section}_mode", "manual")
        section_id = entry.get(f"automated_{section}_prompt_id")
        section_preset = prompt_lookup.get(section_id, {}) if section_id else {}
        section_text = (entry.get(f"manual_{section}") or "").strip()
        if not section_text and section_mode == "automated":
            section_text = (section_preset.get("value") or "").strip()
        sections_metadata[section] = {
            "mode": section_mode,
            "preset_id": section_id,
            "preset_label": section_preset.get("label"),
            "text": section_text,
        }

    return {
        "id": image_id,
        "model": {
            "id": entry.get("model_id"),
            "label": model_cfg["label"] if model_cfg else entry.get("model_label"),
        },
        "prompt": {
            "compiled": entry.get("compiled_prompt", ""),
            "schema_version": entry.get("prompt_schema_version", 1),
            "sections": sections_metadata,
            "prepend": {
                "mode": prepend_mode,
                "preset_id": prepend_id,
                "preset_label": prepend_preset.get("label"),
                "text": prepend_text,
            },
            "main": {
                "mode": prompt_mode,
                "preset_id": automated_id,
                "preset_label": automated_preset.get("label"),
                "text": main_text,
            },
            "append": {
                "mode": append_mode,
                "preset_id": append_id,
                "preset_label": append_preset.get("label"),
                "text": append_text,
            },
        },
        "generation": {
            "width": entry.get("width"),
            "height": entry.get("height"),
            "aspect_ratio": entry.get("aspect_ratio_id"),
            "steps": entry.get("steps"),
            "guidance": entry.get("guidance"),
            "seed": entry.get("seed"),
            "negative_prompt": entry.get("negative_prompt"),
            "scheduler": entry.get("scheduler"),
        },
        "batch": {
            "index": entry.get("batch_index"),
            "size": entry.get("batch_size"),
            "seed": entry.get("batch_seed"),
        },
        "created_at": created_at_iso,
        "is_favourite": entry.get("is_favourite", False),
    }


def _register_generation_cancel_event(generation_id: str | None) -> threading.Event | None:
    """Create and store a cancellation event for a generation request."""
    if not generation_id:
        return None

    if not hasattr(app.state, "generation_cancel_events"):
        app.state.generation_cancel_events = {}

    event = threading.Event()
    app.state.generation_cancel_events[generation_id] = event
    return event


def _pop_generation_cancel_event(generation_id: str | None) -> None:
    """Remove a stored cancellation event after the request finishes."""
    if not generation_id:
        return
    app.state.generation_cancel_events.pop(generation_id, None)


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
    - ``prepend_library`` — prompts loaded from ``prepend.json``.
    - ``main_library`` — prompts loaded from ``main.json``.
    - ``append_library`` — prompts loaded from ``append.json``.
    - ``prepend_prompts`` / ``automated_prompts`` / ``append_prompts`` —
      merged selector catalogs that allow any prompt to be used in any section.

    Returns:
        Dictionary with keys ``version``, ``models``, the three library keys,
        and the three merged selector keys.
    """
    models = _load_json(DATA_DIR / "models.json", {"models": []})
    annotated_models = _annotate_models_with_runtime_support(models.get("models", []))
    prompts = _load_prompt_catalog()
    policy_prompt_options = _load_policy_prompt_options()
    policy_prompt_groups = _load_policy_prompt_groups()
    return {
        "version": __version__,
        "models": annotated_models,
        "prepend_library": prompts.get("prepend_library", []),
        "main_library": prompts.get("main_library", []),
        "append_library": prompts.get("append_library", []),
        "prepend_prompts": prompts.get("prepend_prompts", []),
        "automated_prompts": prompts.get("automated_prompts", []),
        "append_prompts": prompts.get("append_prompts", []),
        "prompt_sections": list(PROMPT_SECTION_ORDER),
        "policy_prompt_options": policy_prompt_options,
        "policy_prompt_groups": policy_prompt_groups,
    }


@app.post("/api/generate")
async def generate_images(req: GenerateRequest) -> dict:
    """Generate a batch of images using the selected diffusion model.

    This endpoint:

    1. Validates the batch size (1–1000).
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
        HTTPException: 400 for invalid batch size, unknown model,
            or unknown prompt mode.
    """
    # --- Validate batch size -----------------------------------------------
    if req.batch_size < 1 or req.batch_size > _MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"batch_size must be between 1 and {_MAX_BATCH_SIZE}",
        )

    # --- Load configuration data -------------------------------------------
    prompts = _load_prompt_catalog()
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

    is_available, unavailable_reason = get_model_runtime_support(model_cfg["hf_id"])
    if not is_available:
        raise HTTPException(
            status_code=503,
            detail=unavailable_reason,
        )

    # --- Resolve prompt input once; placeholders expand per generated image --
    use_section_schema = _request_uses_section_schema(req)
    policy_prompt_options = _load_policy_prompt_options() if use_section_schema else []
    raw_sections = (
        _resolve_structured_prompt_sections(
            req,
            prompts,
            policy_options=policy_prompt_options,
            strict=True,
        )
        if use_section_schema
        else {}
    )
    raw_prepend_value, raw_main_scene, raw_append_value = (
        _resolve_prompt_parts(req, prompts, strict=True) if not use_section_schema else ("", "", "")
    )
    raw_negative_prompt = (req.negative_prompt or "").strip()

    # --- Resolve seed ------------------------------------------------------
    # If the client did not supply a seed, pick a random one.  Each image in
    # the batch gets seed = base_seed + batch_index for reproducibility.
    base_seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

    # --- Ensure model is loaded --------------------------------------------
    model_mgr: ModelManager = app.state.model_manager
    hf_id = model_cfg["hf_id"]

    # --- Register cancellation state ---------------------------------------
    cancel_event = _register_generation_cancel_event(req.generation_id)
    cancelled = False

    try:
        # Switch models if the currently loaded model differs from the request.
        if model_mgr.current_model_id != hf_id:
            await run_in_threadpool(model_mgr.load_model, hf_id)

        # --- Generate batch ------------------------------------------------
        gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
        generated: list[dict] = []
        first_compiled_prompt = ""

        for i in range(req.batch_size):
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break

            # Each image in the batch gets a unique, incrementing seed.
            img_seed = base_seed + i
            img_id = str(uuid.uuid4())
            filename = f"{img_id}.png"
            filepath = GALLERY_DIR / filename
            if use_section_schema:
                section_values = resolve_structured_prompt_variants(raw_sections)
                compiled_prompt = build_structured_prompt(
                    section_values,
                    expand_placeholders=False,
                )
            else:
                prepend_value, main_scene, append_value = resolve_prompt_variants(
                    raw_prepend_value,
                    raw_main_scene,
                    raw_append_value,
                )
                compiled_prompt = build_prompt(
                    prepend_value,
                    main_scene,
                    append_value,
                    prepend_mode=req.prepend_mode,
                    append_mode=req.append_mode,
                    expand_placeholders=False,
                )
            negative_prompt = (
                expand_prompt_placeholders(raw_negative_prompt) if raw_negative_prompt else None
            )
            if not first_compiled_prompt:
                first_compiled_prompt = compiled_prompt

            # Generate the image using the real diffusion pipeline.
            image = await run_in_threadpool(
                model_mgr.generate,
                prompt=compiled_prompt,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance_scale=req.guidance,
                seed=img_seed,
                negative_prompt=negative_prompt,
                scheduler=req.scheduler,
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
                "prompt_schema_version": 2 if use_section_schema else 1,
                "prepend_prompt_id": req.prepend_prompt_id,
                "prompt_mode": req.prompt_mode,
                "manual_prompt": req.manual_prompt,
                "automated_prompt_id": req.automated_prompt_id,
                "append_prompt_id": req.append_prompt_id,
                "prepend_mode": req.prepend_mode,
                "append_mode": req.append_mode,
                "manual_prepend": req.manual_prepend,
                "manual_append": req.manual_append,
                "subject_mode": req.subject_mode,
                "manual_subject": req.manual_subject,
                "automated_subject_prompt_id": req.automated_subject_prompt_id,
                "setting_mode": req.setting_mode,
                "manual_setting": req.manual_setting,
                "automated_setting_prompt_id": req.automated_setting_prompt_id,
                "details_mode": req.details_mode,
                "manual_details": req.manual_details,
                "automated_details_prompt_id": req.automated_details_prompt_id,
                "lighting_mode": req.lighting_mode,
                "manual_lighting": req.manual_lighting,
                "automated_lighting_prompt_id": req.automated_lighting_prompt_id,
                "atmosphere_mode": req.atmosphere_mode,
                "manual_atmosphere": req.manual_atmosphere,
                "automated_atmosphere_prompt_id": req.automated_atmosphere_prompt_id,
                "aspect_ratio_id": req.aspect_ratio_id,
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "guidance": req.guidance,
                "seed": img_seed,
                "negative_prompt": negative_prompt,
                "is_favourite": False,
                "created_at": time.time(),
                "batch_index": i,
                "batch_size": req.batch_size,
                "batch_seed": base_seed,
                "scheduler": req.scheduler,
            }

            # Insert newest first so the gallery is in reverse-chronological order.
            gallery.insert(0, entry)
            generated.append(entry)

        # Persist the updated gallery to disk.
        save_gallery_entries(GALLERY_DB, gallery)
    finally:
        _pop_generation_cancel_event(req.generation_id)

    return {
        "success": True,
        "batch_seed": base_seed,
        "compiled_prompt": first_compiled_prompt,
        "images": generated,
        "cancelled": cancelled,
        "requested_count": req.batch_size,
        "completed_count": len(generated),
    }


@app.post("/api/generate/cancel")
async def cancel_generation(req: CancelGenerationRequest) -> dict:
    """Request cancellation of an in-flight generation batch.

    Cancellation is cooperative: the current image is allowed to finish, then
    the batch loop stops before the next image begins.
    """
    cancel_events: dict[str, threading.Event] = getattr(app.state, "generation_cancel_events", {})
    event = cancel_events.get(req.generation_id)
    if not event:
        raise HTTPException(status_code=404, detail="Generation not found")

    event.set()
    return {"success": True, "generation_id": req.generation_id, "status": "cancelling"}


@app.get("/api/gallery")
async def get_gallery(
    page: int = 1,
    per_page: int = 20,
    favourites_only: bool = False,
    model_id: str | None = None,
) -> dict:
    """Return a paginated listing of gallery images.

    Supports filtering by favourite status and model ID.

    Before filtering, the gallery metadata is reconciled against the gallery
    directory so stale entries caused by manual file deletion do not inflate the
    reported image count, pagination, or filtered totals.

    Args:
        page: Page number (1-indexed).
        per_page: Number of images per page.
        favourites_only: If ``True``, return only favourited images.
        model_id: If provided, return only images from this model.

    Returns:
        Dictionary with keys ``total``, ``page``, ``per_page``, ``pages``,
        and ``images``.
    """
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
    filtered_gallery = filter_gallery_entries(
        gallery,
        favourites_only=favourites_only,
        model_id=model_id,
    )
    return paginate_gallery_entries(filtered_gallery, page, per_page)


@app.get("/api/gallery/runs")
async def get_gallery_runs(
    page: int = 1,
    per_page: int = 20,
    model_id: str | None = None,
    thumbnail_limit: int = 6,
) -> dict:
    """Return gallery images grouped by generation run.

    Images are grouped by ``batch_seed`` into runs, sorted by date
    descending.  Each run includes up to ``thumbnail_limit`` image entries
    for display, plus a ``total_images`` count for the full run.

    Args:
        page: Page number (1-indexed, paginating runs not images).
        per_page: Number of runs per page.
        model_id: Optional model filter.
        thumbnail_limit: Max image entries returned per run.

    Returns:
        Paginated run listing with ``total_runs``, ``total_images``,
        ``page``, ``pages``, and ``runs``.
    """
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
    filtered = filter_gallery_entries(gallery, model_id=model_id)
    runs = group_entries_into_runs(filtered)

    # Truncate each run's images to the thumbnail limit for the response,
    # but preserve the full total_images count.
    for run in runs:
        run["thumbnail_count"] = min(len(run["images"]), thumbnail_limit)
        run["images"] = run["images"][:thumbnail_limit]

    return paginate_runs(runs, page, per_page)


@app.get("/api/gallery/runs/{batch_seed}")
async def get_gallery_run(batch_seed: int) -> dict:
    """Return all images for a specific generation run.

    Args:
        batch_seed: The ``batch_seed`` identifying the generation run.

    Returns:
        Dictionary with ``batch_seed``, ``total_images``, and ``images``.

    Raises:
        HTTPException: 404 if no images match the given batch_seed.
    """
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
    run_entries = get_run_entries(gallery, batch_seed)
    if not run_entries:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "batch_seed": batch_seed,
        "total_images": len(run_entries),
        "images": run_entries,
    }


@app.get("/api/gallery/runs/{batch_seed}/zip")
async def download_run_zip(batch_seed: int) -> Response:
    """Download a zip archive containing all images and metadata for a run.

    The zip contains a flat list of ``pipeworks_{id_short}.png`` and
    ``pipeworks_{id_short}_metadata.json`` pairs — the same format as the
    per-image zip endpoint, but bundled for every image in the run.

    Args:
        batch_seed: The ``batch_seed`` identifying the generation run.

    Returns:
        A ``Response`` with ``application/zip`` content type.

    Raises:
        HTTPException: 404 if no images match the given batch_seed.
    """
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
    run_entries = get_run_entries(gallery, batch_seed)

    if not run_entries:
        raise HTTPException(status_code=404, detail="Run not found")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in run_entries:
            image_path = GALLERY_DIR / entry["filename"]
            if not image_path.exists():
                continue

            id_short = entry["id"][:8]
            metadata = _build_zip_metadata_for_entry(entry)

            zf.write(image_path, f"pipeworks_{id_short}.png")
            zf.writestr(
                f"pipeworks_{id_short}_metadata.json",
                json.dumps(metadata, indent=2),
            )
    buffer.seek(0)

    zip_filename = f"pipeworks_run_{batch_seed}.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


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
    gallery: list[dict] = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
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
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
    entry = next((g for g in gallery if g["id"] == req.image_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")

    # Update the favourite flag and persist to disk.
    entry["is_favourite"] = req.is_favourite
    save_gallery_entries(GALLERY_DB, gallery)

    return {"success": True, "id": req.image_id, "is_favourite": req.is_favourite}


@app.post("/api/gallery/bulk-delete")
async def bulk_delete_images(req: BulkDeleteRequest) -> dict:
    """Delete multiple images from the gallery in a single request.

    For each image ID in the request, removes the PNG file from disk and the
    metadata entry from ``gallery.json``.  IDs that are not found are reported
    in the ``not_found`` list but do not cause the request to fail.

    Args:
        req: Validated :class:`BulkDeleteRequest` payload containing the list
            of image UUIDs to delete.

    Returns:
        Dictionary with ``success``, ``deleted`` (list of removed IDs), and
        ``not_found`` (list of IDs that were not in the gallery).
    """
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)

    # Build a lookup for O(1) access by ID.
    gallery_by_id: dict[str, dict] = {g["id"]: g for g in gallery}

    deleted: list[str] = []
    not_found: list[str] = []

    for image_id in req.image_ids:
        entry = gallery_by_id.get(image_id)
        if not entry:
            not_found.append(image_id)
            continue

        # Remove the PNG file from disk.
        filepath = GALLERY_DIR / entry["filename"]
        if filepath.exists():
            filepath.unlink()

        deleted.append(image_id)

    # Remove all deleted entries from the gallery list and persist.
    deleted_set = set(deleted)
    gallery = [g for g in gallery if g["id"] not in deleted_set]
    save_gallery_entries(GALLERY_DB, gallery)

    return {"success": True, "deleted": deleted, "not_found": not_found}


@app.post("/api/gallery/bulk-zip")
async def bulk_zip_images(req: BulkZipRequest) -> Response:
    """Download a zip archive containing the requested gallery images.

    Args:
        req: Validated :class:`BulkZipRequest` payload containing the list
            of image UUIDs to include.

    Returns:
        A ``Response`` with ``application/zip`` content type.

    Raises:
        HTTPException: 404 if none of the requested images were found.
    """
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
    gallery_by_id: dict[str, dict] = {g["id"]: g for g in gallery}

    buffer = io.BytesIO()
    included = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for image_id in req.image_ids:
            entry = gallery_by_id.get(image_id)
            if not entry:
                continue

            image_path = GALLERY_DIR / entry["filename"]
            if not image_path.exists():
                continue

            id_short = entry["id"][:8]
            metadata = _build_zip_metadata_for_entry(entry)

            zf.write(image_path, f"pipeworks_{id_short}.png")
            zf.writestr(
                f"pipeworks_{id_short}_metadata.json",
                json.dumps(metadata, indent=2),
            )
            included += 1

    if included == 0:
        raise HTTPException(status_code=404, detail="No matching images found")

    buffer.seek(0)
    zip_filename = f"pipeworks_selected_{included}.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


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
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
    entry = next((g for g in gallery if g["id"] == image_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")

    # Remove the PNG file from disk.
    filepath = GALLERY_DIR / entry["filename"]
    if filepath.exists():
        filepath.unlink()

    # Remove the entry from the gallery list and persist.
    gallery = [g for g in gallery if g["id"] != image_id]
    save_gallery_entries(GALLERY_DB, gallery)

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
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
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


@app.get("/api/gallery/{image_id}/zip")
async def download_image_zip(image_id: str) -> Response:
    """Download a zip archive containing the image and structured metadata.

    The zip contains two files:

    - ``pipeworks_{id_short}.png`` — the generated image.
    - ``pipeworks_{id_short}_metadata.json`` — structured metadata with
      prompt sections listed individually (prepend, main, append) plus
      all generation parameters.

    Args:
        image_id: UUID of the gallery image.

    Returns:
        A ``Response`` with ``application/zip`` content type.

    Raises:
        HTTPException: 404 if the image or its PNG file is not found.
    """
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
    entry = next((g for g in gallery if g["id"] == image_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Image not found")

    # Resolve the PNG file on disk.
    image_path = GALLERY_DIR / entry["filename"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    id_short = image_id[:8]
    metadata = _build_zip_metadata_for_entry(entry)

    # Build the zip archive in memory.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(image_path, f"pipeworks_{id_short}.png")
        zf.writestr(
            f"pipeworks_{id_short}_metadata.json",
            json.dumps(metadata, indent=2),
        )
    buffer.seek(0)

    zip_filename = f"pipeworks_{id_short}.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


@app.post("/api/prompt/compile")
async def compile_prompt(req: GenerateRequest) -> dict:
    """Preview the compiled prompt without generating an image.

    Resolves the prepend, main scene, and append parts from the request
    and returns the fully compiled prompt string.

    Args:
        req: Validated :class:`GenerateRequest` payload (only prompt-related
            fields are used).

    Returns:
        Dictionary with ``compiled_prompt`` and ``token_counts``.
    """
    prompts = _load_prompt_catalog()
    models_data = _load_json(DATA_DIR / "models.json", {"models": []})
    use_section_schema = _request_uses_section_schema(req)
    if use_section_schema:
        raw_sections = _resolve_structured_prompt_sections(
            req,
            prompts,
            policy_options=_load_policy_prompt_options(),
            strict=False,
        )
        resolved_sections = resolve_structured_prompt_variants(raw_sections)
        compiled = build_structured_prompt(
            resolved_sections,
            expand_placeholders=False,
        )
    else:
        raw_prepend_value, raw_main_scene, raw_append_value = _resolve_prompt_parts(
            req,
            prompts,
            strict=False,
        )
        prepend_value, main_scene, append_value = resolve_prompt_variants(
            raw_prepend_value,
            raw_main_scene,
            raw_append_value,
        )
        compiled = build_prompt(
            prepend_value,
            main_scene,
            append_value,
            prepend_mode=req.prepend_mode,
            append_mode=req.append_mode,
            expand_placeholders=False,
        )
    model_cfg = next(
        (m for m in models_data.get("models", []) if m["id"] == req.model_id),
        None,
    )
    token_counter: PromptTokenCounter = app.state.prompt_token_counter
    if use_section_schema:
        token_counts = token_counter.count_prompt_sections(
            hf_id=model_cfg.get("hf_id") if model_cfg else None,
            subject_text=resolved_sections.get("subject", ""),
            setting_text=resolved_sections.get("setting", ""),
            details_text=resolved_sections.get("details", ""),
            lighting_text=resolved_sections.get("lighting", ""),
            atmosphere_text=resolved_sections.get("atmosphere", ""),
            compiled_prompt=compiled,
        )
    else:
        token_counts = token_counter.count_prompt_parts(
            hf_id=model_cfg.get("hf_id") if model_cfg else None,
            prepend_text=prepend_value,
            main_text=main_scene,
            append_text=append_value,
            compiled_prompt=compiled,
        )
    return {
        "compiled_prompt": compiled,
        "token_counts": token_counts,
    }


@app.get("/api/stats")
async def get_stats() -> dict:
    """Return gallery statistics.

    Like ``GET /api/gallery``, this endpoint reconciles ``gallery.json`` with
    the gallery directory before counting.  Images that were deleted manually
    from disk therefore stop contributing to the reported totals immediately.

    Returns:
        Dictionary with ``total_images``, ``total_favourites``, and
        ``model_counts`` (a mapping of model_id → count).
    """
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
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
    import copy

    import uvicorn

    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["default"]["fmt"] = "img-gen %(levelprefix)s %(message)s"
    log_config["formatters"]["access"][
        "fmt"
    ] = 'img-gen %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'

    uvicorn.run(
        "pipeworks.api.main:app",
        host=config.server_host,
        port=config.server_port,
        reload=False,
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
