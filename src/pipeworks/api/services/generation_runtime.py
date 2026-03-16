"""Generation runtime tracking and shared payload helpers."""

from __future__ import annotations

import threading
import time
from base64 import b64decode
from typing import TypedDict

from fastapi import HTTPException

from pipeworks.api.models import GenerateRequest
from pipeworks.api.services.prompt_resolution import request_uses_section_schema
from pipeworks.core.config import GpuWorkerConfig, PipeworksConfig


class GenerationJob(TypedDict):
    """Deterministic generation job payload shared by local/remote flows."""

    index: int
    seed: int
    prompt: str
    negative_prompt: str | None


class GenerationRuntimeService:
    """In-memory cancellation and remote-target tracking for active generations."""

    def __init__(self, *, config: PipeworksConfig) -> None:
        self._config = config
        self._generation_cancel_events: dict[str, threading.Event] = {}
        self._worker_cancel_events: dict[str, threading.Event] = {}
        self._remote_generation_targets: dict[str, GpuWorkerConfig] = {}

    def register_generation_cancel_event(self, generation_id: str | None) -> threading.Event | None:
        if not generation_id:
            return None
        event = threading.Event()
        self._generation_cancel_events[generation_id] = event
        return event

    def pop_generation_cancel_event(self, generation_id: str | None) -> None:
        if not generation_id:
            return
        self._generation_cancel_events.pop(generation_id, None)

    def resolve_generation_cancel_event(self, generation_id: str) -> threading.Event | None:
        return self._generation_cancel_events.get(generation_id)

    def register_remote_generation_target(
        self,
        generation_id: str | None,
        worker: GpuWorkerConfig,
    ) -> None:
        if not generation_id:
            return
        self._remote_generation_targets[generation_id] = worker

    def pop_remote_generation_target(self, generation_id: str | None) -> None:
        if not generation_id:
            return
        self._remote_generation_targets.pop(generation_id, None)

    def get_remote_generation_target(self, generation_id: str) -> GpuWorkerConfig | None:
        return self._remote_generation_targets.get(generation_id)

    def register_worker_cancel_event(self, generation_id: str) -> threading.Event:
        event = threading.Event()
        self._worker_cancel_events[generation_id] = event
        return event

    def pop_worker_cancel_event(self, generation_id: str) -> None:
        self._worker_cancel_events.pop(generation_id, None)

    def get_worker_cancel_event(self, generation_id: str) -> threading.Event | None:
        return self._worker_cancel_events.get(generation_id)

    def build_remote_generate_payload(
        self,
        *,
        generation_id: str,
        hf_id: str,
        req: GenerateRequest,
        jobs: list[GenerationJob],
    ) -> dict[str, object]:
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

    def extract_worker_png_bytes(
        self,
        *,
        worker_label: str,
        requested_jobs: list[GenerationJob],
        worker_results: list[dict],
    ) -> dict[int, bytes]:
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
            if decoded_total > self._config.remote_worker_max_decoded_bytes:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Remote worker '{worker_label}' response exceeded decoded size limit "
                        f"({self._config.remote_worker_max_decoded_bytes} bytes)."
                    ),
                )

            decoded_by_index[index] = png_bytes

        return decoded_by_index

    @staticmethod
    def build_gallery_entry(
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
        return {
            "id": image_id,
            "filename": filename,
            "url": f"/static/gallery/{filename}",
            "model_id": req.model_id,
            "model_label": model_cfg["label"],
            "compiled_prompt": compiled_prompt,
            "prompt_schema_version": 2 if request_uses_section_schema(req) else 1,
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
