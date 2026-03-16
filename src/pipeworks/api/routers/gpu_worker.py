"""GPU settings and internal worker API routes."""

from __future__ import annotations

import io
import json
from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from pipeworks.api.models import (
    CancelGenerationRequest,
    GpuSettingsTestRequest,
    GpuSettingsUpdateRequest,
    WorkerGenerateBatchRequest,
)
from pipeworks.api.services.generation_runtime import GenerationRuntimeService
from pipeworks.api.services.gpu_workers import GpuWorkerService
from pipeworks.api.services.http_transport import worker_api_error_detail
from pipeworks.api.services.prompt_catalog import load_json
from pipeworks.core.model_manager import ModelManager

WORKER_GENERATE_BATCH_PATH = "/api/worker/generate-batch"
WORKER_CANCEL_PATH = "/api/worker/generate/cancel"
WORKER_MAX_BATCH_SIZE = 1000


class UrlopenResponse(Protocol):
    """Protocol for urlopen response object used by the GPU health check."""

    def __enter__(self) -> UrlopenResponse: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...

    def read(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class GpuWorkerRouterDependencies:
    """Dependencies required by GPU settings and worker endpoints."""

    data_dir: Callable[[], Path]
    gpu_worker_service: GpuWorkerService
    generation_runtime_service: GenerationRuntimeService
    urlopen: Callable[[UrlRequest, float], UrlopenResponse]


def create_gpu_worker_router(deps: GpuWorkerRouterDependencies) -> APIRouter:
    """Build APIRouter for GPU settings and worker endpoints."""
    router = APIRouter()

    @router.get("/api/gpu-settings")
    async def get_gpu_settings() -> dict:
        return deps.gpu_worker_service.build_gpu_settings_summary()

    @router.post("/api/gpu-settings")
    async def update_gpu_settings(payload: GpuSettingsUpdateRequest) -> dict:
        workers, default_worker_id, generated_token = (
            deps.gpu_worker_service.build_runtime_gpu_workers_from_settings(payload)
        )
        deps.gpu_worker_service.set_runtime_gpu_settings(
            workers=workers,
            default_gpu_worker_id=default_worker_id,
        )
        deps.gpu_worker_service.persist_runtime_gpu_settings(
            workers=workers,
            default_gpu_worker_id=default_worker_id,
        )
        return deps.gpu_worker_service.build_gpu_settings_summary(
            generated_bearer_token=generated_token
        )

    @router.post("/api/gpu-settings/test")
    async def test_gpu_settings_connection(payload: GpuSettingsTestRequest) -> dict:
        base_url = payload.remote_base_url.strip().rstrip("/")
        token = (payload.bearer_token or "").strip()
        if not base_url:
            raise HTTPException(status_code=400, detail="Remote GPU URL is required.")
        if not token:
            matching_remote = next(
                (
                    worker
                    for worker in deps.gpu_worker_service.active_gpu_workers()
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
            with deps.urlopen(  # noqa: S310  # nosec B310
                request, payload.timeout_seconds
            ) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = worker_api_error_detail(exc)
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

    @router.post(WORKER_GENERATE_BATCH_PATH)
    async def worker_generate_batch(req: WorkerGenerateBatchRequest, request: Request) -> dict:
        deps.gpu_worker_service.require_worker_api_auth(request.headers.get("authorization"))

        if len(req.jobs) > WORKER_MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"jobs length must be <= {WORKER_MAX_BATCH_SIZE}",
            )

        models_data = load_json(deps.data_dir() / "models.json", {"models": []})
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

        model_mgr: ModelManager = request.app.state.model_manager
        cancel_event = deps.generation_runtime_service.register_worker_cancel_event(
            req.generation_id
        )
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
            deps.generation_runtime_service.pop_worker_cancel_event(req.generation_id)

        return {
            "success": True,
            "cancelled": cancelled,
            "completed_count": len(results),
            "results": results,
        }

    @router.post(WORKER_CANCEL_PATH)
    async def worker_cancel_generation(req: CancelGenerationRequest, request: Request) -> dict:
        deps.gpu_worker_service.require_worker_api_auth(request.headers.get("authorization"))
        event = deps.generation_runtime_service.get_worker_cancel_event(req.generation_id)
        if not event:
            raise HTTPException(status_code=404, detail="Generation not found")
        event.set()
        return {"success": True, "generation_id": req.generation_id, "status": "cancelling"}

    @router.get("/api/worker/health")
    async def worker_health(request: Request) -> dict:
        deps.gpu_worker_service.require_worker_api_auth(request.headers.get("authorization"))
        model_mgr: ModelManager = request.app.state.model_manager
        return {
            "success": True,
            "status": "ok",
            "loaded_model_hf_id": model_mgr.current_model_id,
        }

    return router
