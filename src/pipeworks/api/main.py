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
import os
import random
import threading
import time
import uuid
import zipfile
from base64 import b64decode, b64encode
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_urlsafe
from threading import RLock
from typing import TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

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
    GpuSettingsTestRequest,
    GpuSettingsUpdateRequest,
    RuntimeAuthResponse,
    RuntimeLoginRequest,
    RuntimeLoginResponse,
    RuntimeLogoutResponse,
    RuntimeModeOptionResponse,
    RuntimeModeRequest,
    RuntimeModeResponse,
    WorkerGenerateBatchRequest,
)
from pipeworks.api.mud_api_client import (
    fetch_mud_api_json,
    fetch_mud_api_json_anonymous,
    normalize_base_url,
)
from pipeworks.api.prompt_builder import (
    SECTION_ORDER,
    build_prompt,
    build_structured_prompt,
    expand_prompt_placeholders,
    resolve_prompt_variants,
    resolve_structured_prompt_variants,
)
from pipeworks.api.runtime_mode import (
    get_runtime_mode,
    set_runtime_mode,
)
from pipeworks.core.config import GpuWorkerConfig, config
from pipeworks.core.model_manager import ModelManager, get_model_runtime_support
from pipeworks.core.prompt_token_counter import PromptTokenCounter

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 1000
_WORKER_MAX_BATCH_SIZE = 1000
_WORKER_GENERATE_BATCH_PATH = "/api/worker/generate-batch"
_WORKER_CANCEL_PATH = "/api/worker/generate/cancel"
PROMPT_SECTION_ORDER = SECTION_ORDER

# ---------------------------------------------------------------------------
# Resolve paths from the global configuration instance.
# ---------------------------------------------------------------------------
STATIC_DIR: Path = config.static_dir
DATA_DIR: Path = config.data_dir
GALLERY_DIR: Path = config.gallery_dir
TEMPLATES_DIR: Path = config.templates_dir
GALLERY_DB: Path = DATA_DIR / "gallery.json"
GPU_SETTINGS_DB: Path = config.outputs_dir / "gpu_workers.runtime.json"

# Ensure the gallery directory exists at import time.
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_MUD_API_BASE_URL = "http://127.0.0.1:8000"
_POLICY_API_ROLE_REQUIRED_DETAIL = "Policy API requires admin or superuser role."
_SNIPPET_ALLOWED_ROLES = {"admin", "superuser"}
_SNIPPET_POLICY_TYPES = {
    "prompt",
    "species_block",
    "image_block",
    "clothing_block",
    "descriptor_layer",
    "registry",
}

_RUNTIME_SESSION_COOKIE_NAME = "pw_image_runtime_session"
_RUNTIME_SESSION_MAX_AGE_SECONDS = 12 * 60 * 60
_DEFAULT_REMOTE_GPU_BASE_URL = "http://100.107.250.105:7860"
_DEFAULT_REMOTE_GPU_LABEL = "Remote GPU (Tailscale)"


@dataclass(slots=True)
class _RuntimeBrowserSession:
    """Server-side runtime session binding for browser refresh persistence."""

    session_id: str
    mode_key: str
    server_url: str
    available_worlds: list[dict[str, object]]
    created_at_epoch: int
    updated_at_epoch: int


@dataclass(frozen=True, slots=True)
class _MudApiRuntimeConfig:
    """Resolved mud-server API runtime configuration."""

    base_url: str
    session_id: str
    timeout_seconds: float = 8.0


runtime_browser_sessions: dict[str, _RuntimeBrowserSession] = {}
runtime_browser_sessions_lock = RLock()


class _GenerationJob(TypedDict):
    """Deterministic generation job payload shared by local/remote flows."""

    index: int
    seed: int
    prompt: str
    negative_prompt: str | None


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
    app.state.worker_cancel_events = {}
    app.state.remote_generation_targets = {}
    app.state.runtime_gpu_workers = None
    app.state.runtime_default_gpu_worker_id = None
    _load_runtime_gpu_settings_from_disk()
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


def _normalize_server_url_for_binding(value: str | None) -> str:
    """Normalize runtime server URLs for stable session-binding comparisons."""
    return str(value or "").strip().rstrip("/")


def _sanitize_available_worlds(
    world_rows: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Retain only stable world-row dictionaries with non-empty IDs."""
    if not isinstance(world_rows, list):
        return []
    sanitized: list[dict[str, object]] = []
    for row in world_rows:
        if not isinstance(row, dict):
            continue
        world_id = str(row.get("id") or "").strip()
        if not world_id:
            continue
        normalized_row = dict(row)
        normalized_row["id"] = world_id
        world_name = str(row.get("name") or "").strip()
        if world_name:
            normalized_row["name"] = world_name
        sanitized.append(normalized_row)
    return sanitized


def _runtime_cookie_secure(request: Request) -> bool:
    """Return whether runtime session cookies should include the Secure attribute."""
    if request.url.scheme == "https":
        return True
    hostname = str(request.url.hostname or "").strip().lower()
    return hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}


def _set_runtime_session_cookie(response: Response, *, request: Request, token: str) -> None:
    """Set hardened browser cookie for one runtime browser-session token."""
    response.set_cookie(
        key=_RUNTIME_SESSION_COOKIE_NAME,
        value=token,
        max_age=_RUNTIME_SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=_runtime_cookie_secure(request),
        samesite="strict",
        path="/",
    )


def _clear_runtime_session_cookie(response: Response, *, request: Request) -> None:
    """Delete runtime session cookie from browser storage."""
    response.delete_cookie(
        key=_RUNTIME_SESSION_COOKIE_NAME,
        httponly=True,
        secure=_runtime_cookie_secure(request),
        samesite="strict",
        path="/",
    )


def _purge_expired_runtime_browser_sessions(*, now_epoch: int | None = None) -> None:
    """Evict expired runtime browser-session records from in-memory store."""
    now = int(now_epoch if now_epoch is not None else time.time())
    with runtime_browser_sessions_lock:
        expired_tokens = [
            token
            for token, record in runtime_browser_sessions.items()
            if now - record.updated_at_epoch >= _RUNTIME_SESSION_MAX_AGE_SECONDS
        ]
        for token in expired_tokens:
            runtime_browser_sessions.pop(token, None)


def _store_runtime_browser_session(
    *,
    mode_key: str,
    server_url: str | None,
    session_id: str,
    available_worlds: list[dict[str, object]] | None,
) -> str:
    """Create one runtime browser-session record and return opaque token."""
    now = int(time.time())
    token = token_urlsafe(32)
    record = _RuntimeBrowserSession(
        session_id=session_id,
        mode_key=mode_key,
        server_url=_normalize_server_url_for_binding(server_url),
        available_worlds=_sanitize_available_worlds(available_worlds),
        created_at_epoch=now,
        updated_at_epoch=now,
    )
    with runtime_browser_sessions_lock:
        runtime_browser_sessions[token] = record
    _purge_expired_runtime_browser_sessions(now_epoch=now)
    return token


def _pop_runtime_browser_session_by_token(token: str | None) -> _RuntimeBrowserSession | None:
    """Remove one runtime browser-session record by token and return it."""
    normalized_token = str(token or "").strip()
    if not normalized_token:
        return None
    with runtime_browser_sessions_lock:
        return runtime_browser_sessions.pop(normalized_token, None)


def _resolve_runtime_browser_session(
    *,
    request: Request,
    mode_key: str,
    server_url: str | None,
) -> tuple[str | None, list[dict[str, object]], str | None]:
    """Resolve runtime browser-session for request cookie and active mode/url."""
    _purge_expired_runtime_browser_sessions()
    token = str(request.cookies.get(_RUNTIME_SESSION_COOKIE_NAME, "")).strip()
    if not token:
        return (None, [], None)
    with runtime_browser_sessions_lock:
        record = runtime_browser_sessions.get(token)
        if record is None:
            return (None, [], token)
        if record.mode_key != mode_key or record.server_url != _normalize_server_url_for_binding(
            server_url
        ):
            runtime_browser_sessions.pop(token, None)
            return (None, [], token)
        record.updated_at_epoch = int(time.time())
        return (
            record.session_id,
            [dict(row) for row in record.available_worlds],
            token,
        )


def _resolve_request_session_id(
    *,
    request: Request,
    mode_key: str,
    server_url: str | None,
    explicit_session_id: str | None,
) -> tuple[str | None, list[dict[str, object]], str | None]:
    """Resolve runtime session from explicit value first, then browser cookie."""
    normalized_explicit = str(explicit_session_id or "").strip()
    if normalized_explicit:
        return (normalized_explicit, [], None)
    return _resolve_runtime_browser_session(
        request=request,
        mode_key=mode_key,
        server_url=server_url,
    )


def _resolve_mud_api_runtime_config(
    *,
    session_id_override: str | None,
    base_url_override: str | None = None,
) -> _MudApiRuntimeConfig:
    """Resolve mud-server runtime config from overrides and defaults."""
    base_url = normalize_base_url(base_url_override or _DEFAULT_MUD_API_BASE_URL)
    if not base_url:
        raise ValueError("Mud API base URL must not be empty.")

    session_id = str(session_id_override or "").strip()
    if not session_id:
        raise ValueError("No active runtime session. Login with an admin/superuser account.")

    return _MudApiRuntimeConfig(base_url=base_url, session_id=session_id, timeout_seconds=8.0)


def _fetch_mud_api_json(
    *,
    runtime: _MudApiRuntimeConfig,
    method: str,
    path: str,
    query_params: dict[str, str],
    json_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Issue one mud-server API request with session query injection."""
    return fetch_mud_api_json(
        runtime=runtime,
        method=method,
        path=path,
        query_params=query_params,
        json_payload=json_payload,
    )


def _fetch_mud_api_json_anonymous(
    *,
    base_url: str,
    method: str,
    path: str,
    body: dict[str, object] | None,
) -> dict[str, object]:
    """Issue one mud-server API request without session query injection."""
    return fetch_mud_api_json_anonymous(
        base_url=base_url,
        method=method,
        path=path,
        body=body,
        timeout_seconds=8.0,
    )


def _extract_available_worlds_from_login_payload(
    payload: dict[str, object],
) -> list[dict[str, object]]:
    """Extract world rows from canonical mud-server ``/login`` payloads."""
    raw_worlds = payload.get("available_worlds")
    if not isinstance(raw_worlds, list):
        return []

    worlds: list[dict[str, object]] = []
    for world in raw_worlds:
        if not isinstance(world, dict):
            continue
        world_id = str(world.get("id") or "").strip()
        if not world_id:
            continue
        normalized_world = dict(world)
        normalized_world["id"] = world_id
        world_name = str(world.get("name") or "").strip()
        if world_name:
            normalized_world["name"] = world_name
        worlds.append(normalized_world)
    return worlds


def _classify_runtime_auth_probe_error(error_detail: str) -> tuple[str, str]:
    """Classify capability probe failures into stable UI-facing auth status."""
    if _POLICY_API_ROLE_REQUIRED_DETAIL in error_detail:
        return ("forbidden", "Session is valid but role is not admin/superuser.")
    if "Invalid or expired session" in error_detail or "Invalid session user" in error_detail:
        return ("unauthenticated", "Session is invalid or expired.")
    return ("error", error_detail)


def _probe_runtime_auth(
    *,
    mode_key: str,
    source_kind: str,
    active_server_url: str | None,
    session_id_override: str | None,
) -> RuntimeAuthResponse:
    """Build runtime auth/capability payload for server-backed snippet loading."""
    if source_kind != "server_api":
        return RuntimeAuthResponse(
            mode_key=mode_key,
            source_kind=source_kind,
            active_server_url=active_server_url,
            session_present=False,
            access_granted=False,
            status="error",
            detail="Runtime mode must be server_api.",
            available_worlds=[],
        )

    try:
        runtime = _resolve_mud_api_runtime_config(
            session_id_override=session_id_override,
            base_url_override=active_server_url,
        )
    except ValueError as exc:
        return RuntimeAuthResponse(
            mode_key=mode_key,
            source_kind=source_kind,
            active_server_url=active_server_url,
            session_present=False,
            access_granted=False,
            status="missing_session",
            detail=str(exc),
            available_worlds=[],
        )

    try:
        _fetch_mud_api_json(
            runtime=runtime,
            method="GET",
            path="/api/policy-capabilities",
            query_params={},
        )
    except ValueError as exc:
        status, detail = _classify_runtime_auth_probe_error(str(exc))
        return RuntimeAuthResponse(
            mode_key=mode_key,
            source_kind=source_kind,
            active_server_url=runtime.base_url,
            session_present=True,
            access_granted=False,
            status=status,
            detail=detail,
            available_worlds=[],
        )

    return RuntimeAuthResponse(
        mode_key=mode_key,
        source_kind=source_kind,
        active_server_url=runtime.base_url,
        session_present=True,
        access_granted=True,
        status="authorized",
        detail="Session is authorized for admin/superuser policy APIs.",
        available_worlds=[],
    )


def _build_runtime_mode_response() -> RuntimeModeResponse:
    """Return runtime mode payload serialized to response models."""
    state = get_runtime_mode()
    return RuntimeModeResponse(
        mode_key=state.mode_key,
        source_kind=state.source_kind,
        active_server_url=state.active_server_url,
        options=[
            RuntimeModeOptionResponse(
                mode_key=option.mode_key,
                label=option.label,
                source_kind=option.source_kind,
                default_server_url=option.default_server_url,
                active_server_url=(
                    state.active_server_url if option.mode_key == state.mode_key else None
                ),
                url_editable=option.url_editable,
            )
            for option in state.options
        ],
    )


def _format_policy_option_label(policy_key: str, variant: str) -> str:
    """Build a stable, human-readable label for one policy snippet option."""
    normalized_key = policy_key.replace("_", " ").replace("-", " ").strip()
    key_label = " ".join(part.capitalize() for part in normalized_key.split()) or policy_key
    variant_label = str(variant or "").strip()
    if variant_label:
        return f"{key_label} ({variant_label})"
    return key_label


def _extract_policy_prompt_text(policy_item: dict[str, object]) -> str:
    """Extract snippet text from one mud-server policy object payload."""
    content = policy_item.get("content")
    if not isinstance(content, dict):
        return ""
    text_value = content.get("text")
    if not isinstance(text_value, str):
        return ""
    return text_value.strip()


def _load_policy_prompt_options(
    *,
    active_server_url: str | None,
    session_id: str | None,
) -> list[dict]:
    """Load prompt snippets from canonical mud-server policy APIs."""
    try:
        runtime = _resolve_mud_api_runtime_config(
            session_id_override=session_id,
            base_url_override=active_server_url,
        )
    except ValueError:
        return []

    try:
        payload = _fetch_mud_api_json(
            runtime=runtime,
            method="GET",
            path="/api/policies",
            query_params={},
        )
    except ValueError as exc:
        logger.warning("Unable to load policy snippets from mud-server API: %s", exc)
        return []

    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return []

    options: list[dict] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        policy_type = str(item.get("policy_type") or "").strip()
        if policy_type not in _SNIPPET_POLICY_TYPES:
            continue

        text_value = _extract_policy_prompt_text(item)
        if not text_value:
            continue

        policy_id = str(item.get("policy_id") or "").strip()
        variant = str(item.get("variant") or "").strip()
        policy_key = str(item.get("policy_key") or "").strip()
        namespace = str(item.get("namespace") or "").strip()
        if not policy_id or not variant or not policy_key:
            continue

        option_id = f"{policy_id}:{variant}"
        group = namespace or policy_type
        options.append(
            {
                "id": option_id,
                "label": _format_policy_option_label(policy_key, variant),
                "value": text_value,
                "group": group,
                "path": option_id,
            }
        )

    options.sort(key=lambda option: (option.get("group", ""), option.get("label", "")))
    return options


def _load_policy_prompt_groups(options: list[dict]) -> list[str]:
    """Load snippet group labels mirrored from canonical policy namespaces."""
    return sorted(
        {str(option.get("group") or "").strip() for option in options if option.get("group")}
    )


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
    runtime_state = get_runtime_mode()
    policy_options = _load_policy_prompt_options(
        active_server_url=runtime_state.active_server_url,
        session_id=None,
    )
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


def _public_gpu_worker(worker: GpuWorkerConfig) -> dict[str, object]:
    """Return a worker payload safe for public API responses."""
    return {
        "id": worker.id,
        "label": worker.label,
        "mode": worker.mode,
        "enabled": worker.enabled,
    }


def _resolve_default_gpu_worker_id_or_error(
    workers: list[GpuWorkerConfig],
    preferred_worker_id: str | None,
) -> str:
    """Resolve default worker ID for a worker list, validating enabled state."""
    if not workers:
        raise ValueError("At least one GPU worker must be configured.")

    enabled_workers = [worker for worker in workers if worker.enabled]
    if not enabled_workers:
        raise ValueError("At least one GPU worker must be enabled.")

    preferred = (preferred_worker_id or "").strip()
    if not preferred:
        return enabled_workers[0].id

    target = next((worker for worker in workers if worker.id == preferred), None)
    if target is None:
        raise ValueError(
            f"default_gpu_worker_id '{preferred}' does not match any configured worker."
        )
    if not target.enabled:
        raise ValueError(f"default_gpu_worker_id '{preferred}' must reference an enabled worker.")
    return preferred


def _active_gpu_workers() -> list[GpuWorkerConfig]:
    """Return active GPU worker configuration (runtime override or static config)."""
    runtime_workers = getattr(app.state, "runtime_gpu_workers", None)
    if isinstance(runtime_workers, list) and runtime_workers:
        return runtime_workers
    return config.gpu_workers


def _active_default_gpu_worker_id() -> str:
    """Return active default GPU worker ID."""
    runtime_default = getattr(app.state, "runtime_default_gpu_worker_id", None)
    fallback_default = (
        runtime_default if runtime_default is not None else config.default_gpu_worker_id
    )
    return _resolve_default_gpu_worker_id_or_error(_active_gpu_workers(), fallback_default)


def _active_worker_api_tokens() -> set[str]:
    """Return accepted worker API tokens, including runtime worker overrides."""
    tokens = set(config.worker_api_tokens())
    for worker in _active_gpu_workers():
        if worker.mode == "remote" and worker.bearer_token:
            tokens.add(worker.bearer_token.strip())
    return {token for token in tokens if token}


def _set_runtime_gpu_settings(
    *,
    workers: list[GpuWorkerConfig],
    default_gpu_worker_id: str | None,
) -> None:
    """Apply runtime GPU worker overrides to in-memory app state."""
    resolved_default = _resolve_default_gpu_worker_id_or_error(workers, default_gpu_worker_id)
    app.state.runtime_gpu_workers = workers
    app.state.runtime_default_gpu_worker_id = resolved_default


def _persist_runtime_gpu_settings(
    *,
    workers: list[GpuWorkerConfig],
    default_gpu_worker_id: str,
) -> None:
    """Persist runtime GPU worker overrides to disk."""
    payload = {
        "gpu_workers": [worker.model_dump() for worker in workers],
        "default_gpu_worker_id": default_gpu_worker_id,
    }
    GPU_SETTINGS_DB.parent.mkdir(parents=True, exist_ok=True)
    GPU_SETTINGS_DB.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.chmod(GPU_SETTINGS_DB, 0o600)


def _load_runtime_gpu_settings_from_disk() -> None:
    """Load persisted runtime GPU worker settings when available."""
    if not GPU_SETTINGS_DB.exists():
        return

    try:
        current_mode = GPU_SETTINGS_DB.stat().st_mode & 0o777
        if current_mode != 0o600:
            os.chmod(GPU_SETTINGS_DB, 0o600)
        parsed = json.loads(GPU_SETTINGS_DB.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("GPU settings payload must be an object.")
        raw_workers = parsed.get("gpu_workers")
        if not isinstance(raw_workers, list):
            raise ValueError("gpu_workers must be a list.")
        workers = [GpuWorkerConfig.model_validate(item) for item in raw_workers]
        raw_default = parsed.get("default_gpu_worker_id")
        default_gpu_worker_id = raw_default if isinstance(raw_default, str) else None
        _set_runtime_gpu_settings(
            workers=workers,
            default_gpu_worker_id=default_gpu_worker_id,
        )
    except Exception:
        logger.exception("Failed to load runtime GPU settings from %s.", GPU_SETTINGS_DB)


def _build_gpu_settings_summary(
    *,
    generated_bearer_token: str | None = None,
) -> dict[str, object]:
    """Return UI summary payload for GPU settings panel."""
    workers = _active_gpu_workers()
    default_worker_id = _active_default_gpu_worker_id()
    remote_worker = next((worker for worker in workers if worker.mode == "remote"), None)
    summary: dict[str, object] = {
        "use_remote_gpu": remote_worker is not None and remote_worker.enabled,
        "remote_label": remote_worker.label if remote_worker else _DEFAULT_REMOTE_GPU_LABEL,
        "remote_base_url": (
            remote_worker.base_url if remote_worker else _DEFAULT_REMOTE_GPU_BASE_URL
        ),
        "remote_timeout_seconds": (remote_worker.timeout_seconds if remote_worker else 240.0),
        "has_bearer_token": bool(remote_worker and remote_worker.bearer_token),
        "default_gpu_worker_id": default_worker_id,
    }
    if generated_bearer_token:
        summary["generated_bearer_token"] = generated_bearer_token
    return summary


def _build_runtime_gpu_workers_from_settings(
    payload: GpuSettingsUpdateRequest,
) -> tuple[list[GpuWorkerConfig], str, str | None]:
    """Build active worker list/default from one GPU settings update request."""
    active_workers = _active_gpu_workers()
    local_worker = next((worker for worker in active_workers if worker.mode == "local"), None)
    if local_worker is None:
        local_worker = GpuWorkerConfig(
            id="local",
            label="Local GPU",
            mode="local",
            enabled=True,
        )
    local_worker = local_worker.model_copy(update={"enabled": True})

    existing_remote = next((worker for worker in active_workers if worker.mode == "remote"), None)

    if not payload.use_remote_gpu:
        workers = [local_worker]
        default_id = local_worker.id
        resolved_default = _resolve_default_gpu_worker_id_or_error(workers, default_id)
        return workers, resolved_default, None

    base_url = (
        (payload.remote_base_url or "").strip()
        or (existing_remote.base_url if existing_remote else "")
        or _DEFAULT_REMOTE_GPU_BASE_URL
    )
    if not base_url:
        raise HTTPException(status_code=400, detail="Remote GPU URL is required.")

    resolved_token = (payload.bearer_token or "").strip()
    if not resolved_token and existing_remote and existing_remote.bearer_token:
        resolved_token = existing_remote.bearer_token.strip()

    generated_token: str | None = None
    if not resolved_token:
        generated_token = token_urlsafe(32)
        resolved_token = generated_token

    remote_label = (
        (payload.remote_label or "").strip()
        or (existing_remote.label if existing_remote else "")
        or _DEFAULT_REMOTE_GPU_LABEL
    )
    remote_worker = GpuWorkerConfig(
        id=(existing_remote.id if existing_remote else "remote-ts"),
        label=remote_label,
        mode="remote",
        base_url=base_url,
        bearer_token=resolved_token,
        timeout_seconds=payload.timeout_seconds,
        enabled=True,
    )
    workers = [local_worker, remote_worker]
    default_id = remote_worker.id if payload.default_to_remote else local_worker.id
    resolved_default = _resolve_default_gpu_worker_id_or_error(workers, default_id)
    return workers, resolved_default, generated_token


def _resolve_gpu_worker_or_400(worker_id: str | None) -> GpuWorkerConfig:
    """Resolve requested/default worker, rejecting invalid or disabled targets."""
    selected_id = (worker_id or _active_default_gpu_worker_id()).strip()
    target = next((worker for worker in _active_gpu_workers() if worker.id == selected_id), None)
    if target is None:
        raise HTTPException(status_code=400, detail=f"Unknown gpu_worker_id: {selected_id}")
    if not target.enabled:
        raise HTTPException(
            status_code=400,
            detail=f"GPU worker '{target.label}' is disabled.",
        )
    return target


def _worker_api_error_detail(exc: HTTPError) -> str:
    """Extract best-effort detail from worker HTTP error payload."""
    default_detail = f"HTTP {exc.code}"
    try:
        raw = exc.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        parsed = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return default_detail

    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if detail:
            return str(detail)
    return default_detail


def _post_json_with_bearer(
    *,
    base_url: str,
    path: str,
    bearer_token: str,
    timeout_seconds: float,
    payload: dict[str, object],
) -> dict[str, object]:
    """POST JSON with bearer auth and decode object responses."""
    url = f"{base_url.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    request = UrlRequest(
        url=url,
        method="POST",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310  # nosec B310
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ValueError(f"{path} failed: {_worker_api_error_detail(exc)}") from exc
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} failed: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{path} response must be a JSON object.")
    return parsed


def _extract_worker_png_bytes(
    *,
    worker_label: str,
    requested_jobs: list[_GenerationJob],
    worker_results: list[dict],
) -> dict[int, bytes]:
    """Validate and decode worker PNG results, enforcing response size limits."""
    requested_indexes = {job["index"] for job in requested_jobs}
    decoded_total = 0
    decoded_by_index: dict[int, bytes] = {}

    if len(worker_results) > len(requested_jobs):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Remote worker '{worker_label}' returned too many results "
                f"({len(worker_results)} > {len(requested_jobs)})."
            ),
        )

    for item in worker_results:
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=502,
                detail=f"Remote worker '{worker_label}' returned an invalid result record.",
            )

        index = item.get("index")
        seed = item.get("seed")
        png_base64 = item.get("png_base64")

        if not isinstance(index, int) or index not in requested_indexes:
            raise HTTPException(
                status_code=502,
                detail=f"Remote worker '{worker_label}' returned an unexpected result index.",
            )
        if not isinstance(seed, int):
            raise HTTPException(
                status_code=502,
                detail=f"Remote worker '{worker_label}' returned an invalid seed value.",
            )
        if not isinstance(png_base64, str) or not png_base64:
            raise HTTPException(
                status_code=502,
                detail=f"Remote worker '{worker_label}' returned invalid PNG payload data.",
            )
        if index in decoded_by_index:
            raise HTTPException(
                status_code=502,
                detail=f"Remote worker '{worker_label}' returned duplicate image indexes.",
            )

        try:
            png_bytes = b64decode(png_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Remote worker '{worker_label}' returned invalid base64 image data.",
            ) from exc

        decoded_total += len(png_bytes)
        if decoded_total > config.remote_worker_max_decoded_bytes:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Remote worker '{worker_label}' response exceeded decoded size limit "
                    f"({config.remote_worker_max_decoded_bytes} bytes)."
                ),
            )

        decoded_by_index[index] = png_bytes

    return decoded_by_index


def _build_remote_generate_payload(
    *,
    generation_id: str,
    hf_id: str,
    req: GenerateRequest,
    jobs: list[_GenerationJob],
) -> dict[str, object]:
    """Build the controller -> worker generate-batch payload."""
    return {
        "generation_id": generation_id,
        "hf_id": hf_id,
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "guidance": req.guidance,
        "scheduler": req.scheduler,
        "jobs": jobs,
    }


def _build_gallery_entry(
    *,
    req: GenerateRequest,
    model_cfg: dict,
    image_id: str,
    filename: str,
    compiled_prompt: str,
    negative_prompt: str | None,
    seed: int,
    batch_index: int,
    batch_seed: int,
    compute_target_id: str,
    compute_target_label: str,
) -> dict[str, object]:
    """Build one persisted gallery metadata record for a generated image."""
    return {
        "id": image_id,
        "filename": filename,
        "url": f"/static/gallery/{filename}",
        "model_id": req.model_id,
        "model_label": model_cfg["label"],
        "compiled_prompt": compiled_prompt,
        "prompt_schema_version": 2 if _request_uses_section_schema(req) else 1,
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
        "seed": seed,
        "negative_prompt": negative_prompt,
        "is_favourite": False,
        "created_at": time.time(),
        "batch_index": batch_index,
        "batch_size": req.batch_size,
        "batch_seed": batch_seed,
        "scheduler": req.scheduler,
        "compute_target_id": compute_target_id,
        "compute_target_label": compute_target_label,
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


def _register_remote_generation_target(
    generation_id: str | None,
    worker: GpuWorkerConfig,
) -> None:
    """Track the selected remote worker for one in-flight generation request."""
    if not generation_id:
        return
    if not hasattr(app.state, "remote_generation_targets"):
        app.state.remote_generation_targets = {}
    app.state.remote_generation_targets[generation_id] = worker


def _pop_remote_generation_target(generation_id: str | None) -> None:
    """Remove tracked remote target metadata for an in-flight generation."""
    if not generation_id:
        return
    if not hasattr(app.state, "remote_generation_targets"):
        return
    app.state.remote_generation_targets.pop(generation_id, None)


def _get_remote_generation_target(generation_id: str) -> GpuWorkerConfig | None:
    """Return tracked remote worker metadata for a generation id."""
    remote_targets: dict[str, GpuWorkerConfig] = getattr(app.state, "remote_generation_targets", {})
    return remote_targets.get(generation_id)


def _register_worker_cancel_event(generation_id: str) -> threading.Event:
    """Create and store a worker-side cancellation event."""
    if not hasattr(app.state, "worker_cancel_events"):
        app.state.worker_cancel_events = {}
    event = threading.Event()
    app.state.worker_cancel_events[generation_id] = event
    return event


def _pop_worker_cancel_event(generation_id: str) -> None:
    """Remove a worker-side cancellation event."""
    if not hasattr(app.state, "worker_cancel_events"):
        return
    app.state.worker_cancel_events.pop(generation_id, None)


def _require_worker_api_auth(request: Request) -> None:
    """Validate bearer token for internal worker API endpoints."""
    expected_tokens = _active_worker_api_tokens()
    auth_header = str(request.headers.get("authorization", "")).strip()
    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        raise HTTPException(status_code=401, detail="Worker API requires bearer token.")

    presented = auth_header[len(prefix) :].strip()
    if not presented or presented not in expected_tokens:
        raise HTTPException(status_code=401, detail="Invalid worker API bearer token.")


def _pop_generation_cancel_event(generation_id: str | None) -> None:
    """Remove a stored cancellation event after the request finishes."""
    if not generation_id:
        return
    app.state.generation_cancel_events.pop(generation_id, None)


def _load_policy_prompts_for_request(
    *,
    request: Request,
    response: Response | None = None,
    explicit_session_id: str | None = None,
) -> tuple[list[dict], list[str], RuntimeAuthResponse]:
    """Resolve auth context and return canonical policy snippet options."""
    state = get_runtime_mode()
    resolved_session_id, cookie_worlds, cookie_token = _resolve_request_session_id(
        request=request,
        mode_key=state.mode_key,
        server_url=state.active_server_url,
        explicit_session_id=explicit_session_id,
    )

    runtime_auth = _probe_runtime_auth(
        mode_key=state.mode_key,
        source_kind=state.source_kind,
        active_server_url=state.active_server_url,
        session_id_override=resolved_session_id,
    )

    if cookie_token and runtime_auth.status in {"missing_session", "unauthenticated"}:
        _pop_runtime_browser_session_by_token(cookie_token)
        if response is not None:
            _clear_runtime_session_cookie(response, request=request)

    if runtime_auth.access_granted and cookie_worlds:
        runtime_auth = runtime_auth.model_copy(update={"available_worlds": cookie_worlds})

    if not runtime_auth.access_granted:
        return ([], [], runtime_auth)

    policy_prompt_options = _load_policy_prompt_options(
        active_server_url=state.active_server_url,
        session_id=resolved_session_id,
    )
    policy_prompt_groups = _load_policy_prompt_groups(policy_prompt_options)
    return (policy_prompt_options, policy_prompt_groups, runtime_auth)


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


@app.get("/api/runtime-mode", response_model=RuntimeModeResponse)
async def api_runtime_mode() -> RuntimeModeResponse:
    """Return active runtime mode and available source-mode profiles."""
    return _build_runtime_mode_response()


@app.post("/api/runtime-mode", response_model=RuntimeModeResponse)
async def api_runtime_mode_set(payload: RuntimeModeRequest) -> RuntimeModeResponse:
    """Switch active runtime mode and optional mud-server URL override."""
    try:
        set_runtime_mode(mode_key=payload.mode_key, server_url=payload.server_url)
        return _build_runtime_mode_response()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runtime-auth", response_model=RuntimeAuthResponse)
async def api_runtime_auth(
    request: Request,
    response: Response,
    session_id: str | None = Query(default=None),
) -> RuntimeAuthResponse:
    """Return runtime auth/access status for server-backed snippet APIs."""
    _, _, runtime_auth = _load_policy_prompts_for_request(
        request=request,
        response=response,
        explicit_session_id=session_id,
    )
    return runtime_auth


@app.post("/api/runtime-login", response_model=RuntimeLoginResponse)
async def api_runtime_login(
    payload: RuntimeLoginRequest,
    request: Request,
    response: Response,
) -> RuntimeLoginResponse:
    """Authenticate to active mud-server profile and return session bootstrap data."""
    state = get_runtime_mode()
    if state.source_kind != "server_api":
        raise HTTPException(status_code=400, detail="Runtime mode must be server_api.")

    username = (payload.username or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required.")
    password = (payload.password or "").strip()
    if not password:
        raise HTTPException(status_code=400, detail="Password is required.")

    base_url = normalize_base_url(
        state.active_server_url
        if state.active_server_url is not None
        else _DEFAULT_MUD_API_BASE_URL
    )
    if not base_url:
        raise HTTPException(status_code=400, detail="Mud API base URL must not be empty.")

    try:
        login_payload = _fetch_mud_api_json_anonymous(
            base_url=base_url,
            method="POST",
            path="/login",
            body={"username": username, "password": password},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = login_payload.get("session_id")
    role = str(login_payload.get("role") or "").strip()
    available_worlds = _sanitize_available_worlds(
        _extract_available_worlds_from_login_payload(login_payload)
    )

    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(
            status_code=400, detail="Mud login response did not include session_id."
        )
    if not role:
        raise HTTPException(status_code=400, detail="Mud login response did not include role.")

    success = role in _SNIPPET_ALLOWED_ROLES
    detail = (
        "Authenticated as admin/superuser."
        if success
        else "Authenticated, but role is not admin/superuser for policy APIs."
    )

    if success:
        token = _store_runtime_browser_session(
            mode_key=state.mode_key,
            server_url=state.active_server_url,
            session_id=session_id.strip(),
            available_worlds=available_worlds,
        )
        _set_runtime_session_cookie(response, request=request, token=token)
    else:
        stale_token = str(request.cookies.get(_RUNTIME_SESSION_COOKIE_NAME, "")).strip()
        if stale_token:
            _pop_runtime_browser_session_by_token(stale_token)
        _clear_runtime_session_cookie(response, request=request)

    return RuntimeLoginResponse(
        success=success,
        session_id=None,
        role=role,
        available_worlds=available_worlds,
        detail=detail,
    )


@app.post("/api/runtime-logout", response_model=RuntimeLogoutResponse)
async def api_runtime_logout(request: Request, response: Response) -> RuntimeLogoutResponse:
    """Clear active browser-bound runtime session token."""
    session_token = str(request.cookies.get(_RUNTIME_SESSION_COOKIE_NAME, "")).strip()
    if session_token:
        _pop_runtime_browser_session_by_token(session_token)
    _clear_runtime_session_cookie(response, request=request)
    return RuntimeLogoutResponse(success=True, detail="Runtime session cleared.")


@app.get("/api/policy-prompts")
async def get_policy_prompts(
    request: Request,
    response: Response,
    session_id: str | None = Query(default=None),
) -> dict:
    """Return canonical policy snippet options for prompt composer dropdowns."""
    options, groups, runtime_auth = _load_policy_prompts_for_request(
        request=request,
        response=response,
        explicit_session_id=session_id,
    )
    return {
        "policy_prompt_options": options,
        "policy_prompt_groups": groups,
        "runtime_auth": runtime_auth.model_dump(),
    }


@app.get("/api/config")
async def get_config(request: Request, response: Response) -> dict:
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
    policy_prompt_options, policy_prompt_groups, runtime_auth = _load_policy_prompts_for_request(
        request=request,
        response=response,
    )
    return {
        "version": __version__,
        "models": annotated_models,
        "gpu_workers": [_public_gpu_worker(worker) for worker in _active_gpu_workers()],
        "default_gpu_worker_id": _active_default_gpu_worker_id(),
        "prepend_library": prompts.get("prepend_library", []),
        "main_library": prompts.get("main_library", []),
        "append_library": prompts.get("append_library", []),
        "prepend_prompts": prompts.get("prepend_prompts", []),
        "automated_prompts": prompts.get("automated_prompts", []),
        "append_prompts": prompts.get("append_prompts", []),
        "prompt_sections": list(PROMPT_SECTION_ORDER),
        "policy_prompt_options": policy_prompt_options,
        "policy_prompt_groups": policy_prompt_groups,
        "runtime_mode": _build_runtime_mode_response().model_dump(),
        "runtime_auth": runtime_auth.model_dump(),
    }


@app.get("/api/gpu-settings")
async def get_gpu_settings() -> dict:
    """Return editable GPU settings summary for the frontend admin panel."""
    return _build_gpu_settings_summary()


@app.post("/api/gpu-settings")
async def update_gpu_settings(payload: GpuSettingsUpdateRequest) -> dict:
    """Update runtime GPU settings and persist them to local disk."""
    workers, default_worker_id, generated_token = _build_runtime_gpu_workers_from_settings(payload)
    _set_runtime_gpu_settings(
        workers=workers,
        default_gpu_worker_id=default_worker_id,
    )
    _persist_runtime_gpu_settings(
        workers=workers,
        default_gpu_worker_id=default_worker_id,
    )
    return _build_gpu_settings_summary(generated_bearer_token=generated_token)


@app.post("/api/gpu-settings/test")
async def test_gpu_settings_connection(payload: GpuSettingsTestRequest) -> dict:
    """Probe remote worker health using supplied URL/token credentials."""
    base_url = payload.remote_base_url.strip().rstrip("/")
    token = (payload.bearer_token or "").strip()
    if not base_url:
        raise HTTPException(status_code=400, detail="Remote GPU URL is required.")
    if not token:
        matching_remote = next(
            (
                worker
                for worker in _active_gpu_workers()
                if worker.mode == "remote" and (worker.base_url or "").rstrip("/") == base_url
            ),
            None,
        )
        if matching_remote and matching_remote.bearer_token:
            token = matching_remote.bearer_token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Bearer token is required.")

    request = UrlRequest(
        url=f"{base_url}/api/worker/health",
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urlopen(
            request, timeout=payload.timeout_seconds
        ) as response:  # noqa: S310  # nosec B310
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = _worker_api_error_detail(exc)
        raise HTTPException(
            status_code=502, detail=f"Remote GPU health check failed: {detail}"
        ) from exc
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Remote GPU health check failed: {exc}",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502, detail="Remote GPU health response must be JSON object."
        )

    return {
        "success": True,
        "detail": "Remote GPU health check succeeded.",
        "worker": parsed,
    }


@app.post("/api/generate")
async def generate_images(req: GenerateRequest, request: Request) -> dict:
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

    target_worker = _resolve_gpu_worker_or_400(req.gpu_worker_id)
    if target_worker.mode == "remote" and req.batch_size > config.remote_worker_max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"batch_size {req.batch_size} exceeds remote worker limit "
                f"({config.remote_worker_max_batch_size})."
            ),
        )

    # --- Load configuration data -------------------------------------------
    prompts = _load_prompt_catalog()
    models_data = _load_json(DATA_DIR / "models.json", {"models": []})

    # --- Resolve model configuration ---------------------------------------
    model_cfg = next(
        (m for m in models_data.get("models", []) if m["id"] == req.model_id),
        None,
    )
    if not model_cfg:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {req.model_id}",
        )

    hf_id = model_cfg["hf_id"]
    if target_worker.mode == "local":
        is_available, unavailable_reason = get_model_runtime_support(hf_id)
        if not is_available:
            raise HTTPException(
                status_code=503,
                detail=unavailable_reason,
            )

    # --- Resolve prompt input once; placeholders expand per generated image --
    use_section_schema = _request_uses_section_schema(req)
    policy_prompt_options: list[dict] = []
    if use_section_schema:
        policy_prompt_options, _, _ = _load_policy_prompts_for_request(
            request=request,
            response=None,
        )
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
    base_seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

    # --- Build deterministic generation jobs -------------------------------
    jobs: list[_GenerationJob] = []
    for i in range(req.batch_size):
        img_seed = base_seed + i
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
        jobs.append(
            {
                "index": i,
                "seed": img_seed,
                "prompt": compiled_prompt,
                "negative_prompt": negative_prompt,
            }
        )

    cancel_event = _register_generation_cancel_event(req.generation_id)
    gallery = load_gallery_entries(GALLERY_DB, GALLERY_DIR)
    generated: list[dict] = []
    cancelled = False
    remote_generation_id = req.generation_id or f"remote-{uuid.uuid4()}"

    try:
        if target_worker.mode == "local":
            model_mgr: ModelManager = app.state.model_manager
            if model_mgr.current_model_id != hf_id:
                await run_in_threadpool(model_mgr.load_model, hf_id)

            for job in jobs:
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    break

                image = await run_in_threadpool(
                    model_mgr.generate,
                    prompt=job["prompt"],
                    width=req.width,
                    height=req.height,
                    steps=req.steps,
                    guidance_scale=req.guidance,
                    seed=job["seed"],
                    negative_prompt=job["negative_prompt"],
                    scheduler=req.scheduler,
                )

                image_id = str(uuid.uuid4())
                filename = f"{image_id}.png"
                filepath = GALLERY_DIR / filename
                image.save(filepath, format="PNG")

                entry = _build_gallery_entry(
                    req=req,
                    model_cfg=model_cfg,
                    image_id=image_id,
                    filename=filename,
                    compiled_prompt=job["prompt"],
                    negative_prompt=job["negative_prompt"],
                    seed=job["seed"],
                    batch_index=job["index"],
                    batch_seed=base_seed,
                    compute_target_id=target_worker.id,
                    compute_target_label=target_worker.label,
                )
                gallery.insert(0, entry)
                generated.append(entry)
        else:
            _register_remote_generation_target(req.generation_id, target_worker)
            worker_payload = _build_remote_generate_payload(
                generation_id=remote_generation_id,
                hf_id=hf_id,
                req=req,
                jobs=jobs,
            )
            try:
                worker_response = _post_json_with_bearer(
                    base_url=target_worker.base_url or "",
                    path=_WORKER_GENERATE_BATCH_PATH,
                    bearer_token=target_worker.bearer_token or "",
                    timeout_seconds=target_worker.timeout_seconds,
                    payload=worker_payload,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"GPU worker '{target_worker.label}' failed: {exc}",
                ) from exc

            if worker_response.get("success") is not True:
                worker_detail = (
                    worker_response.get("detail") or "worker returned unsuccessful status."
                )
                raise HTTPException(
                    status_code=502,
                    detail=f"GPU worker '{target_worker.label}' failed: {worker_detail}",
                )

            raw_results = worker_response.get("results")
            if not isinstance(raw_results, list):
                raise HTTPException(
                    status_code=502,
                    detail=f"GPU worker '{target_worker.label}' returned invalid results payload.",
                )

            decoded_by_index = _extract_worker_png_bytes(
                worker_label=target_worker.label,
                requested_jobs=jobs,
                worker_results=raw_results,
            )

            for job in jobs:
                job_index = job["index"]
                png_bytes = decoded_by_index.get(job_index)
                if png_bytes is None:
                    continue

                image_id = str(uuid.uuid4())
                filename = f"{image_id}.png"
                filepath = GALLERY_DIR / filename
                filepath.write_bytes(png_bytes)

                entry = _build_gallery_entry(
                    req=req,
                    model_cfg=model_cfg,
                    image_id=image_id,
                    filename=filename,
                    compiled_prompt=job["prompt"],
                    negative_prompt=job["negative_prompt"],
                    seed=job["seed"],
                    batch_index=job_index,
                    batch_seed=base_seed,
                    compute_target_id=target_worker.id,
                    compute_target_label=target_worker.label,
                )
                gallery.insert(0, entry)
                generated.append(entry)

            cancelled = bool(worker_response.get("cancelled"))

        save_gallery_entries(GALLERY_DB, gallery)
    finally:
        _pop_generation_cancel_event(req.generation_id)
        _pop_remote_generation_target(req.generation_id)

    first_compiled_prompt = generated[0]["compiled_prompt"] if generated else ""
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
    remote_target = _get_remote_generation_target(req.generation_id)
    if not event and not remote_target:
        raise HTTPException(status_code=404, detail="Generation not found")

    if event:
        event.set()

    status = "cancelling"
    if remote_target and remote_target.mode == "remote":
        try:
            cancel_payload = _post_json_with_bearer(
                base_url=remote_target.base_url or "",
                path=_WORKER_CANCEL_PATH,
                bearer_token=remote_target.bearer_token or "",
                timeout_seconds=remote_target.timeout_seconds,
                payload={"generation_id": req.generation_id},
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Failed to forward cancellation to GPU worker "
                    f"'{remote_target.label}': {exc}"
                ),
            ) from exc

        if cancel_payload.get("success") is not True:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Failed to forward cancellation to GPU worker " f"'{remote_target.label}'."
                ),
            )
        status = str(cancel_payload.get("status") or status)

    return {"success": True, "generation_id": req.generation_id, "status": status}


@app.post(_WORKER_GENERATE_BATCH_PATH)
async def worker_generate_batch(req: WorkerGenerateBatchRequest, request: Request) -> dict:
    """Internal worker endpoint: generate a batch and return PNG payloads."""
    _require_worker_api_auth(request)

    if len(req.jobs) > _WORKER_MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"jobs length must be <= {_WORKER_MAX_BATCH_SIZE}",
        )

    models_data = _load_json(DATA_DIR / "models.json", {"models": []})
    allowed_hf_ids = {
        model.get("hf_id")
        for model in models_data.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("hf_id"), str)
    }
    if req.hf_id not in allowed_hf_ids:
        raise HTTPException(
            status_code=400,
            detail=f"HF model '{req.hf_id}' is not allowed by worker configuration.",
        )

    model_mgr: ModelManager = app.state.model_manager
    cancel_event = _register_worker_cancel_event(req.generation_id)
    results: list[dict[str, object]] = []
    cancelled = False

    try:
        if model_mgr.current_model_id != req.hf_id:
            await run_in_threadpool(model_mgr.load_model, req.hf_id)

        for job in req.jobs:
            if cancel_event.is_set():
                cancelled = True
                break

            image = await run_in_threadpool(
                model_mgr.generate,
                prompt=job.prompt,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance_scale=req.guidance,
                seed=job.seed,
                negative_prompt=job.negative_prompt,
                scheduler=req.scheduler,
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            results.append(
                {
                    "index": job.index,
                    "seed": job.seed,
                    "png_base64": b64encode(buffer.getvalue()).decode("utf-8"),
                }
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _pop_worker_cancel_event(req.generation_id)

    return {
        "success": True,
        "cancelled": cancelled,
        "completed_count": len(results),
        "results": results,
    }


@app.post(_WORKER_CANCEL_PATH)
async def worker_cancel_generation(req: CancelGenerationRequest, request: Request) -> dict:
    """Internal worker endpoint: cooperatively cancel an active generation."""
    _require_worker_api_auth(request)
    cancel_events: dict[str, threading.Event] = getattr(app.state, "worker_cancel_events", {})
    event = cancel_events.get(req.generation_id)
    if not event:
        raise HTTPException(status_code=404, detail="Generation not found")
    event.set()
    return {"success": True, "generation_id": req.generation_id, "status": "cancelling"}


@app.get("/api/worker/health")
async def worker_health(request: Request) -> dict:
    """Internal worker endpoint: readiness probe."""
    _require_worker_api_auth(request)
    model_mgr: ModelManager = app.state.model_manager
    return {
        "success": True,
        "status": "ok",
        "loaded_model_hf_id": model_mgr.current_model_id,
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
async def compile_prompt(req: GenerateRequest, request: Request) -> dict:
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
        policy_prompt_options, _, _ = _load_policy_prompts_for_request(
            request=request,
            response=None,
        )
        raw_sections = _resolve_structured_prompt_sections(
            req,
            prompts,
            policy_options=policy_prompt_options,
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
