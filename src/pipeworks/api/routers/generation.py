"""Generation and cancellation API routes."""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from pipeworks.api.gallery_store import load_gallery_entries, save_gallery_entries
from pipeworks.api.models import CancelGenerationRequest, GenerateRequest
from pipeworks.api.prompt_builder import (
    build_structured_prompt,
    expand_prompt_placeholders,
    resolve_structured_prompt_variants,
)
from pipeworks.api.routers.gpu_worker import WORKER_CANCEL_PATH, WORKER_GENERATE_BATCH_PATH
from pipeworks.api.services.generation_runtime import GenerationJob, GenerationRuntimeService
from pipeworks.api.services.gpu_workers import GpuWorkerService
from pipeworks.api.services.prompt_catalog import load_json, load_prompt_catalog
from pipeworks.api.services.prompt_resolution import (
    ensure_prompt_schema_v2,
    resolve_structured_prompt_sections,
)
from pipeworks.api.services.runtime_policy import RuntimePolicyService
from pipeworks.core.model_manager import ModelManager


@dataclass(frozen=True, slots=True)
class GenerationRouterDependencies:
    """Dependencies required by generation routes."""

    data_dir: Callable[[], Path]
    gallery_dir: Callable[[], Path]
    gallery_db: Callable[[], Path]
    max_batch_size: int
    gpu_worker_service: GpuWorkerService
    runtime_policy_service: RuntimePolicyService
    generation_runtime_service: GenerationRuntimeService
    normalize_base_url: Callable[[str | None], str]
    post_json_with_bearer: Callable[..., dict[str, object]]
    get_model_runtime_support: Callable[[str], tuple[bool, str | None]]


def create_generation_router(deps: GenerationRouterDependencies) -> APIRouter:
    """Build APIRouter for generation and cancel endpoints."""
    router = APIRouter()

    @router.post("/api/generate")
    async def generate_images(req: GenerateRequest, request: Request) -> dict:
        if req.batch_size < 1 or req.batch_size > deps.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"batch_size must be between 1 and {deps.max_batch_size}",
            )

        ensure_prompt_schema_v2(req)

        target_worker = deps.gpu_worker_service.resolve_gpu_worker_or_400(req.gpu_worker_id)
        generation_id = req.generation_id
        if (
            target_worker.mode == "remote"
            and req.batch_size > deps.gpu_worker_service.remote_worker_max_batch_size
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"batch_size {req.batch_size} exceeds remote worker limit "
                    f"({deps.gpu_worker_service.remote_worker_max_batch_size})."
                ),
            )

        data_dir = deps.data_dir()
        gallery_dir = deps.gallery_dir()
        gallery_db = deps.gallery_db()

        prompts = load_prompt_catalog(data_dir=data_dir)
        models_data = load_json(data_dir / "models.json", {"models": []})

        model_cfg = next(
            (m for m in models_data.get("models", []) if m["id"] == req.model_id), None
        )
        if not model_cfg:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {req.model_id}",
            )

        hf_id = model_cfg["hf_id"]
        if target_worker.mode == "local":
            is_available, unavailable_reason = deps.get_model_runtime_support(hf_id)
            if not is_available:
                raise HTTPException(
                    status_code=503,
                    detail=unavailable_reason,
                )

        policy_prompt_options, _, _ = deps.runtime_policy_service.load_policy_prompts_for_request(
            request=request,
            response=None,
            explicit_session_id=None,
            normalize_base_url=deps.normalize_base_url,
        )
        raw_sections = resolve_structured_prompt_sections(
            req,
            prompts,
            policy_options=policy_prompt_options,
            strict=True,
        )
        raw_negative_prompt = (req.negative_prompt or "").strip()

        base_seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

        jobs: list[GenerationJob] = []
        for index in range(req.batch_size):
            image_seed = base_seed + index
            section_values = resolve_structured_prompt_variants(raw_sections)
            compiled_prompt = build_structured_prompt(
                section_values,
                expand_placeholders=False,
            )
            negative_prompt = (
                expand_prompt_placeholders(raw_negative_prompt) if raw_negative_prompt else None
            )
            jobs.append(
                {
                    "index": index,
                    "seed": image_seed,
                    "prompt": compiled_prompt,
                    "negative_prompt": negative_prompt,
                }
            )

        cancel_event = deps.generation_runtime_service.register_generation_cancel_event(
            generation_id
        )
        gallery = load_gallery_entries(gallery_db, gallery_dir)
        generated: list[dict] = []
        cancelled = False
        remote_generation_id = generation_id or f"remote-{uuid.uuid4()}"

        try:
            if target_worker.mode == "local":
                model_mgr: ModelManager = request.app.state.model_manager
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
                    filepath = gallery_dir / filename
                    image.save(filepath, format="PNG")

                    entry = deps.generation_runtime_service.build_gallery_entry(
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
                deps.generation_runtime_service.register_remote_generation_target(
                    generation_id, target_worker
                )
                worker_payload = deps.generation_runtime_service.build_remote_generate_payload(
                    generation_id=remote_generation_id,
                    hf_id=hf_id,
                    req=req,
                    jobs=jobs,
                )
                try:
                    worker_response = deps.post_json_with_bearer(
                        base_url=target_worker.base_url or "",
                        path=WORKER_GENERATE_BATCH_PATH,
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
                        detail=(
                            f"GPU worker '{target_worker.label}' returned invalid results payload."
                        ),
                    )

                decoded_by_index = deps.generation_runtime_service.extract_worker_png_bytes(
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
                    filepath = gallery_dir / filename
                    filepath.write_bytes(png_bytes)

                    entry = deps.generation_runtime_service.build_gallery_entry(
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

            save_gallery_entries(gallery_db, gallery)
        finally:
            deps.generation_runtime_service.pop_generation_cancel_event(generation_id)
            deps.generation_runtime_service.pop_remote_generation_target(generation_id)

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

    @router.post("/api/generate/cancel")
    async def cancel_generation(req: CancelGenerationRequest) -> dict:
        event = deps.generation_runtime_service.resolve_generation_cancel_event(req.generation_id)
        remote_target = deps.generation_runtime_service.get_remote_generation_target(
            req.generation_id
        )
        if not event and not remote_target:
            raise HTTPException(status_code=404, detail="Generation not found")

        if event:
            event.set()

        status = "cancelling"
        if remote_target and remote_target.mode == "remote":
            try:
                cancel_payload = deps.post_json_with_bearer(
                    base_url=remote_target.base_url or "",
                    path=WORKER_CANCEL_PATH,
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
                    detail=f"Failed to forward cancellation to GPU worker '{remote_target.label}'.",
                )
            status = str(cancel_payload.get("status") or status)

        return {"success": True, "generation_id": req.generation_id, "status": status}

    return router
