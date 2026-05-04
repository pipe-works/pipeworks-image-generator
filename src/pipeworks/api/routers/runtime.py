"""Runtime mode/auth/config API routes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response

from pipeworks import __version__
from pipeworks.api.models import (
    RuntimeAuthResponse,
    RuntimeLoginRequest,
    RuntimeLoginResponse,
    RuntimeLogoutResponse,
    RuntimeModeOptionResponse,
    RuntimeModeRequest,
    RuntimeModeResponse,
)
from pipeworks.api.runtime_mode import get_runtime_mode, set_runtime_mode
from pipeworks.api.services.gpu_workers import GpuWorkerService
from pipeworks.api.services.prompt_catalog import (
    annotate_models_with_runtime_support,
    load_json,
    load_prompt_catalog,
)
from pipeworks.api.services.runtime_policy import RuntimePolicyService


@dataclass(frozen=True, slots=True)
class RuntimeRouterDependencies:
    """Dependencies required by runtime/auth/config routes."""

    data_dir: Callable[[], Path]
    default_mud_api_base_url: str
    normalize_base_url: Callable[[str | None], str]
    runtime_policy_service: RuntimePolicyService
    gpu_worker_service: GpuWorkerService


def _build_runtime_mode_response() -> RuntimeModeResponse:
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


def create_runtime_router(deps: RuntimeRouterDependencies) -> APIRouter:
    """Build APIRouter for runtime mode/auth/config endpoints."""
    router = APIRouter()

    @router.get("/api/runtime-mode", response_model=RuntimeModeResponse)
    async def api_runtime_mode() -> RuntimeModeResponse:
        return _build_runtime_mode_response()

    @router.post("/api/runtime-mode", response_model=RuntimeModeResponse)
    async def api_runtime_mode_set(payload: RuntimeModeRequest) -> RuntimeModeResponse:
        try:
            set_runtime_mode(mode_key=payload.mode_key, server_url=payload.server_url)
            return _build_runtime_mode_response()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/runtime-auth", response_model=RuntimeAuthResponse)
    async def api_runtime_auth(
        request: Request,
        response: Response,
        session_id: str | None = Query(default=None),
    ) -> RuntimeAuthResponse:
        _, _, runtime_auth = deps.runtime_policy_service.load_policy_prompts_for_request(
            request=request,
            response=response,
            explicit_session_id=session_id,
            normalize_base_url=deps.normalize_base_url,
        )
        return runtime_auth

    @router.post("/api/runtime-login", response_model=RuntimeLoginResponse)
    async def api_runtime_login(
        payload: RuntimeLoginRequest,
        request: Request,
        response: Response,
    ) -> RuntimeLoginResponse:
        state = get_runtime_mode()
        if state.source_kind != "server_api":
            raise HTTPException(status_code=400, detail="Runtime mode must be server_api.")

        username = (payload.username or "").strip()
        if not username:
            raise HTTPException(status_code=400, detail="Username is required.")
        password = (payload.password or "").strip()
        if not password:
            raise HTTPException(status_code=400, detail="Password is required.")

        base_url = deps.normalize_base_url(
            state.active_server_url
            if state.active_server_url is not None
            else deps.default_mud_api_base_url
        )
        if not base_url:
            raise HTTPException(status_code=400, detail="Mud API base URL must not be empty.")

        try:
            login_payload = deps.runtime_policy_service.fetch_mud_api_json_anonymous(
                base_url=base_url,
                method="POST",
                path="/login",
                body={"username": username, "password": password},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        session_id = login_payload.get("session_id")
        role = str(login_payload.get("role") or "").strip()
        available_worlds = deps.runtime_policy_service.sanitize_available_worlds(
            deps.runtime_policy_service.extract_available_worlds_from_login_payload(login_payload)
        )

        if not isinstance(session_id, str) or not session_id.strip():
            raise HTTPException(
                status_code=400, detail="Mud login response did not include session_id."
            )
        if not role:
            raise HTTPException(status_code=400, detail="Mud login response did not include role.")

        success = role in deps.runtime_policy_service.snippet_allowed_roles
        detail = (
            "Authenticated as admin/superuser."
            if success
            else "Authenticated, but role is not admin/superuser for policy APIs."
        )

        if success:
            token = deps.runtime_policy_service.login_runtime_session(
                mode_key=state.mode_key,
                server_url=state.active_server_url,
                session_id=session_id.strip(),
                available_worlds=available_worlds,
            )
            deps.runtime_policy_service.set_runtime_session_cookie(
                response, request=request, token=token
            )
        else:
            deps.runtime_policy_service.clear_cookie_session_from_request(request=request)
            deps.runtime_policy_service.clear_runtime_session_cookie(response, request=request)

        return RuntimeLoginResponse(
            success=success,
            session_id=None,
            role=role,
            available_worlds=available_worlds,
            detail=detail,
        )

    @router.post("/api/runtime-logout", response_model=RuntimeLogoutResponse)
    async def api_runtime_logout(request: Request, response: Response) -> RuntimeLogoutResponse:
        deps.runtime_policy_service.clear_cookie_session_from_request(request=request)
        deps.runtime_policy_service.clear_runtime_session_cookie(response, request=request)
        return RuntimeLogoutResponse(success=True, detail="Runtime session cleared.")

    @router.get("/api/policy-prompts")
    async def get_policy_prompts(
        request: Request,
        response: Response,
        session_id: str | None = Query(default=None),
    ) -> dict:
        options, groups, runtime_auth = deps.runtime_policy_service.load_policy_prompts_for_request(
            request=request,
            response=response,
            explicit_session_id=session_id,
            normalize_base_url=deps.normalize_base_url,
        )
        slot_kinds = deps.runtime_policy_service.load_policy_prompt_slot_kinds(options)
        return {
            "policy_prompt_options": options,
            "policy_prompt_groups": groups,
            "policy_prompt_slot_kinds": slot_kinds,
            "runtime_auth": runtime_auth.model_dump(),
        }

    @router.get("/api/config")
    async def get_config(request: Request, response: Response) -> dict:
        data_dir = deps.data_dir()
        models = load_json(data_dir / "models.json", {"models": []})
        annotated_models = annotate_models_with_runtime_support(models.get("models", []))
        prompts = load_prompt_catalog(data_dir=data_dir)
        policy_prompt_options, policy_prompt_groups, runtime_auth = (
            deps.runtime_policy_service.load_policy_prompts_for_request(
                request=request,
                response=response,
                explicit_session_id=None,
                normalize_base_url=deps.normalize_base_url,
            )
        )
        return {
            "version": __version__,
            "models": annotated_models,
            "gpu_workers": [
                deps.gpu_worker_service.public_gpu_worker(worker)
                for worker in deps.gpu_worker_service.active_gpu_workers()
            ],
            "default_gpu_worker_id": deps.gpu_worker_service.active_default_gpu_worker_id(),
            "prepend_library": prompts.get("prepend_library", []),
            "main_library": prompts.get("main_library", []),
            "append_library": prompts.get("append_library", []),
            "prepend_prompts": prompts.get("prepend_prompts", []),
            "automated_prompts": prompts.get("automated_prompts", []),
            "append_prompts": prompts.get("append_prompts", []),
            "prompt_sections": ["subject", "setting", "details", "lighting", "atmosphere"],
            "policy_prompt_options": policy_prompt_options,
            "policy_prompt_groups": policy_prompt_groups,
            "policy_prompt_slot_kinds": deps.runtime_policy_service.load_policy_prompt_slot_kinds(
                policy_prompt_options
            ),
            "runtime_mode": _build_runtime_mode_response().model_dump(),
            "runtime_auth": runtime_auth.model_dump(),
        }

    return router
