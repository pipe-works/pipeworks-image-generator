"""Pipe-Works Image Generator FastAPI application bootstrap."""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pipeworks import __version__
from pipeworks.api.mud_api_client import (
    fetch_mud_api_json,
    fetch_mud_api_json_anonymous,
    normalize_base_url,
)
from pipeworks.api.routers.gallery import GalleryRouterDependencies, create_gallery_router
from pipeworks.api.routers.generation import GenerationRouterDependencies, create_generation_router
from pipeworks.api.routers.gpu_worker import GpuWorkerRouterDependencies, create_gpu_worker_router
from pipeworks.api.routers.prompt import PromptRouterDependencies, create_prompt_router
from pipeworks.api.routers.runtime import RuntimeRouterDependencies, create_runtime_router
from pipeworks.api.services.generation_runtime import GenerationRuntimeService
from pipeworks.api.services.gpu_workers import GpuWorkerService
from pipeworks.api.services.http_transport import post_json_with_bearer
from pipeworks.api.services.prompt_catalog import load_prompt_catalog
from pipeworks.api.services.runtime_policy import RuntimePolicyService
from pipeworks.core.config import config
from pipeworks.core.model_manager import ModelManager
from pipeworks.core.model_manager import get_model_runtime_support as _core_model_runtime_support
from pipeworks.core.prompt_token_counter import PromptTokenCounter

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 1000
_DEFAULT_MUD_API_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_REMOTE_GPU_BASE_URL = "http://100.107.250.105:7860"
_DEFAULT_REMOTE_GPU_LABEL = "Remote GPU (Tailscale)"

STATIC_DIR: Path = config.static_dir
DATA_DIR: Path = config.data_dir
GALLERY_DIR: Path = config.gallery_dir
TEMPLATES_DIR: Path = config.templates_dir
GALLERY_DB: Path = DATA_DIR / "gallery.json"
GPU_SETTINGS_DB: Path = config.outputs_dir / "gpu_workers.runtime.json"

GALLERY_DIR.mkdir(parents=True, exist_ok=True)


def _urlopen_with_timeout(request, timeout):
    """Issue urllib request with explicit timeout for remote worker health checks."""
    return urlopen(request, timeout=timeout)  # noqa: S310  # nosec B310


def _fetch_mud_api_json(
    *,
    runtime,
    method: str,
    path: str,
    query_params: dict[str, str],
    json_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Compatibility wrapper retained for current tests and route patch points."""
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
    """Compatibility wrapper retained for runtime login transport."""
    return fetch_mud_api_json_anonymous(
        base_url=base_url,
        method=method,
        path=path,
        body=body,
        timeout_seconds=8.0,
    )


def _post_json_with_bearer(
    *,
    base_url: str,
    path: str,
    bearer_token: str,
    timeout_seconds: float,
    payload: dict[str, object],
) -> dict[str, object]:
    """Compatibility wrapper retained for current tests and route patch points."""
    return post_json_with_bearer(
        base_url=base_url,
        path=path,
        bearer_token=bearer_token,
        timeout_seconds=timeout_seconds,
        payload=payload,
    )


def get_model_runtime_support(hf_id: str) -> tuple[bool, str | None]:
    """Compatibility wrapper retained for current tests and patch points."""
    return _core_model_runtime_support(hf_id)


gpu_worker_service = GpuWorkerService(
    config=config,
    gpu_settings_db=lambda: GPU_SETTINGS_DB,
    default_remote_gpu_base_url=_DEFAULT_REMOTE_GPU_BASE_URL,
    default_remote_gpu_label=_DEFAULT_REMOTE_GPU_LABEL,
)
runtime_policy_service = RuntimePolicyService(
    config=config,
    default_mud_api_base_url=_DEFAULT_MUD_API_BASE_URL,
    # Keep lambdas so monkeypatching module globals in tests still works.
    fetch_mud_api_json=lambda **kwargs: _fetch_mud_api_json(**kwargs),
    fetch_mud_api_json_anonymous=lambda **kwargs: _fetch_mud_api_json_anonymous(**kwargs),
)
generation_runtime_service = GenerationRuntimeService(config=config)
runtime_browser_sessions = runtime_policy_service.runtime_browser_sessions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle."""
    app.state.config = config
    app.state.model_manager = ModelManager(config)
    app.state.prompt_token_counter = PromptTokenCounter(config)
    app.state.gpu_worker_service = gpu_worker_service
    app.state.runtime_policy_service = runtime_policy_service
    app.state.generation_runtime_service = generation_runtime_service
    gpu_worker_service.load_runtime_gpu_settings_from_disk()

    # Warm prompt catalog to surface any legacy fallback warning at startup.
    load_prompt_catalog(data_dir=DATA_DIR)

    logger.info("ModelManager initialised (no model loaded yet).")

    yield

    app.state.model_manager.unload()
    logger.info("ModelManager unloaded on shutdown.")


app = FastAPI(
    title="Pipe-Works Image Generator",
    description="Image generation API with multi-model diffusion pipeline support.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_http_cache_for_local_testing(request, call_next):
    """Optionally disable browser caching for local development."""
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


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the main application HTML page."""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    raise HTTPException(status_code=404, detail="index.html not found")


_RUNTIME_DEPS = RuntimeRouterDependencies(
    data_dir=lambda: DATA_DIR,
    default_mud_api_base_url=_DEFAULT_MUD_API_BASE_URL,
    normalize_base_url=normalize_base_url,
    runtime_policy_service=runtime_policy_service,
    gpu_worker_service=gpu_worker_service,
)

_GPU_DEPS = GpuWorkerRouterDependencies(
    data_dir=lambda: DATA_DIR,
    gpu_worker_service=gpu_worker_service,
    generation_runtime_service=generation_runtime_service,
    urlopen=_urlopen_with_timeout,
)

_GENERATION_DEPS = GenerationRouterDependencies(
    data_dir=lambda: DATA_DIR,
    gallery_dir=lambda: GALLERY_DIR,
    gallery_db=lambda: GALLERY_DB,
    max_batch_size=_MAX_BATCH_SIZE,
    gpu_worker_service=_RUNTIME_DEPS.gpu_worker_service,
    runtime_policy_service=_RUNTIME_DEPS.runtime_policy_service,
    generation_runtime_service=_GPU_DEPS.generation_runtime_service,
    normalize_base_url=normalize_base_url,
    post_json_with_bearer=lambda **kwargs: _post_json_with_bearer(**kwargs),
    get_model_runtime_support=lambda hf_id: get_model_runtime_support(hf_id),
)

_GALLERY_DEPS = GalleryRouterDependencies(
    data_dir=lambda: DATA_DIR,
    gallery_dir=lambda: GALLERY_DIR,
    gallery_db=lambda: GALLERY_DB,
    runtime_policy_service=_RUNTIME_DEPS.runtime_policy_service,
    normalize_base_url=normalize_base_url,
)

_PROMPT_DEPS = PromptRouterDependencies(
    data_dir=lambda: DATA_DIR,
    runtime_policy_service=_RUNTIME_DEPS.runtime_policy_service,
    normalize_base_url=normalize_base_url,
)

app.include_router(create_runtime_router(_RUNTIME_DEPS))
app.include_router(create_gpu_worker_router(_GPU_DEPS))
app.include_router(create_generation_router(_GENERATION_DEPS))
app.include_router(create_gallery_router(_GALLERY_DEPS))
app.include_router(create_prompt_router(_PROMPT_DEPS))


def main() -> None:
    """Launch the uvicorn ASGI server."""
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
