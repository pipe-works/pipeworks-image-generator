"""Runtime GPU worker settings and auth helpers."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException

from pipeworks.api.models import GpuSettingsUpdateRequest
from pipeworks.core.config import GpuWorkerConfig, PipeworksConfig

logger = logging.getLogger(__name__)


class GpuWorkerService:
    """Manage active GPU worker configuration and runtime overrides."""

    def __init__(
        self,
        *,
        config: PipeworksConfig,
        gpu_settings_db: Path | Callable[[], Path],
        default_remote_gpu_base_url: str,
        default_remote_gpu_label: str,
    ) -> None:
        self._config = config
        self._gpu_settings_db = gpu_settings_db
        self._default_remote_gpu_base_url = default_remote_gpu_base_url
        self._default_remote_gpu_label = default_remote_gpu_label

        self._runtime_gpu_workers: list[GpuWorkerConfig] | None = None
        self._runtime_default_gpu_worker_id: str | None = None

    def _gpu_settings_path(self) -> Path:
        if callable(self._gpu_settings_db):
            return self._gpu_settings_db()
        return self._gpu_settings_db

    @staticmethod
    def public_gpu_worker(worker: GpuWorkerConfig) -> dict[str, object]:
        return {
            "id": worker.id,
            "label": worker.label,
            "mode": worker.mode,
            "enabled": worker.enabled,
        }

    @staticmethod
    def resolve_default_gpu_worker_id_or_error(
        workers: list[GpuWorkerConfig],
        preferred_worker_id: str | None,
    ) -> str:
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
            raise ValueError(
                f"default_gpu_worker_id '{preferred}' must reference an enabled worker."
            )
        return preferred

    def active_gpu_workers(self) -> list[GpuWorkerConfig]:
        if isinstance(self._runtime_gpu_workers, list) and self._runtime_gpu_workers:
            return self._runtime_gpu_workers
        return self._config.gpu_workers

    def active_default_gpu_worker_id(self) -> str:
        fallback_default = (
            self._runtime_default_gpu_worker_id
            if self._runtime_default_gpu_worker_id is not None
            else self._config.default_gpu_worker_id
        )
        return self.resolve_default_gpu_worker_id_or_error(
            self.active_gpu_workers(), fallback_default
        )

    def active_worker_api_tokens(self) -> set[str]:
        tokens = set(self._config.worker_api_tokens())
        for worker in self.active_gpu_workers():
            if worker.mode == "remote" and worker.bearer_token:
                tokens.add(worker.bearer_token.strip())
        return {token for token in tokens if token}

    def set_runtime_gpu_settings(
        self,
        *,
        workers: list[GpuWorkerConfig],
        default_gpu_worker_id: str | None,
    ) -> None:
        resolved_default = self.resolve_default_gpu_worker_id_or_error(
            workers, default_gpu_worker_id
        )
        self._runtime_gpu_workers = workers
        self._runtime_default_gpu_worker_id = resolved_default

    def persist_runtime_gpu_settings(
        self,
        *,
        workers: list[GpuWorkerConfig],
        default_gpu_worker_id: str,
    ) -> None:
        payload = {
            "gpu_workers": [worker.model_dump() for worker in workers],
            "default_gpu_worker_id": default_gpu_worker_id,
        }
        gpu_settings_db = self._gpu_settings_path()
        gpu_settings_db.parent.mkdir(parents=True, exist_ok=True)
        gpu_settings_db.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.chmod(gpu_settings_db, 0o600)

    def load_runtime_gpu_settings_from_disk(self) -> None:
        gpu_settings_db = self._gpu_settings_path()
        if not gpu_settings_db.exists():
            return

        try:
            current_mode = gpu_settings_db.stat().st_mode & 0o777
            if current_mode != 0o600:
                os.chmod(gpu_settings_db, 0o600)
            parsed = json.loads(gpu_settings_db.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("GPU settings payload must be an object.")
            raw_workers = parsed.get("gpu_workers")
            if not isinstance(raw_workers, list):
                raise ValueError("gpu_workers must be a list.")
            workers = [GpuWorkerConfig.model_validate(item) for item in raw_workers]
            raw_default = parsed.get("default_gpu_worker_id")
            default_gpu_worker_id = raw_default if isinstance(raw_default, str) else None
            self.set_runtime_gpu_settings(
                workers=workers,
                default_gpu_worker_id=default_gpu_worker_id,
            )
        except Exception:
            logger.exception("Failed to load runtime GPU settings from %s.", gpu_settings_db)

    def build_gpu_settings_summary(
        self,
        *,
        generated_bearer_token: str | None = None,
    ) -> dict[str, object]:
        workers = self.active_gpu_workers()
        default_worker_id = self.active_default_gpu_worker_id()
        remote_worker = next((worker for worker in workers if worker.mode == "remote"), None)
        summary: dict[str, object] = {
            "use_remote_gpu": remote_worker is not None and remote_worker.enabled,
            "remote_label": (
                remote_worker.label if remote_worker else self._default_remote_gpu_label
            ),
            "remote_base_url": (
                remote_worker.base_url if remote_worker else self._default_remote_gpu_base_url
            ),
            "remote_timeout_seconds": (remote_worker.timeout_seconds if remote_worker else 240.0),
            "has_bearer_token": bool(remote_worker and remote_worker.bearer_token),
            "default_gpu_worker_id": default_worker_id,
        }
        if generated_bearer_token:
            summary["generated_bearer_token"] = generated_bearer_token
        return summary

    def build_runtime_gpu_workers_from_settings(
        self,
        payload: GpuSettingsUpdateRequest,
    ) -> tuple[list[GpuWorkerConfig], str, str | None]:
        active_workers = self.active_gpu_workers()
        local_worker = next((worker for worker in active_workers if worker.mode == "local"), None)
        if local_worker is None:
            local_worker = GpuWorkerConfig(
                id="local",
                label="Luminal GPU",
                mode="local",
                enabled=True,
            )
        local_worker = local_worker.model_copy(update={"enabled": True})

        existing_remote = next(
            (worker for worker in active_workers if worker.mode == "remote"), None
        )

        if not payload.use_remote_gpu:
            workers = [local_worker]
            default_id = local_worker.id
            resolved_default = self.resolve_default_gpu_worker_id_or_error(workers, default_id)
            return workers, resolved_default, None

        base_url = (
            (payload.remote_base_url or "").strip()
            or (existing_remote.base_url if existing_remote else "")
            or self._default_remote_gpu_base_url
        )
        if not base_url:
            raise HTTPException(status_code=400, detail="Remote GPU URL is required.")

        resolved_token = (payload.bearer_token or "").strip()
        if not resolved_token and existing_remote and existing_remote.bearer_token:
            resolved_token = existing_remote.bearer_token.strip()

        generated_token: str | None = None
        if not resolved_token:
            from secrets import token_urlsafe

            generated_token = token_urlsafe(32)
            resolved_token = generated_token

        remote_label = (
            (payload.remote_label or "").strip()
            or (existing_remote.label if existing_remote else "")
            or self._default_remote_gpu_label
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
        resolved_default = self.resolve_default_gpu_worker_id_or_error(workers, default_id)
        return workers, resolved_default, generated_token

    def resolve_gpu_worker_or_400(self, worker_id: str | None) -> GpuWorkerConfig:
        selected_id = (worker_id or self.active_default_gpu_worker_id()).strip()
        target = next(
            (worker for worker in self.active_gpu_workers() if worker.id == selected_id), None
        )
        if target is None:
            raise HTTPException(status_code=400, detail=f"Unknown gpu_worker_id: {selected_id}")
        if not target.enabled:
            raise HTTPException(
                status_code=400,
                detail=f"GPU worker '{target.label}' is disabled.",
            )
        return target

    def require_worker_api_auth(self, auth_header: str | None) -> None:
        expected_tokens = self.active_worker_api_tokens()
        normalized_header = str(auth_header or "").strip()
        prefix = "Bearer "
        if not normalized_header.startswith(prefix):
            raise HTTPException(status_code=401, detail="Worker API requires bearer token.")

        presented = normalized_header[len(prefix) :].strip()
        if not presented or presented not in expected_tokens:
            raise HTTPException(status_code=401, detail="Invalid worker API bearer token.")

    @property
    def remote_worker_max_batch_size(self) -> int:
        """Expose remote worker batch-size cap for route validation."""
        return self._config.remote_worker_max_batch_size
