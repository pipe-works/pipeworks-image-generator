"""Prompt compile preview API routes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Request

from pipeworks.api.models import GenerateRequest
from pipeworks.api.prompt_builder import (
    build_dynamic_prompt,
    build_structured_prompt,
    resolve_dynamic_prompt_variants,
    resolve_structured_prompt_variants,
)
from pipeworks.api.services.prompt_catalog import load_json, load_prompt_catalog
from pipeworks.api.services.prompt_resolution import (
    ensure_prompt_schema,
    resolve_dynamic_prompt_sections,
    resolve_structured_prompt_sections,
)
from pipeworks.api.services.runtime_policy import RuntimePolicyService
from pipeworks.core.prompt_token_counter import PromptTokenCounter


@dataclass(frozen=True, slots=True)
class PromptRouterDependencies:
    """Dependencies required by prompt compile endpoint."""

    data_dir: Callable[[], Path]
    runtime_policy_service: RuntimePolicyService
    normalize_base_url: Callable[[str | None], str]


def create_prompt_router(deps: PromptRouterDependencies) -> APIRouter:
    """Build APIRouter for prompt compile endpoint."""
    router = APIRouter()

    @router.post("/api/prompt/compile")
    async def compile_prompt(req: GenerateRequest, request: Request) -> dict:
        schema_version = ensure_prompt_schema(req)
        data_dir = deps.data_dir()
        prompts = load_prompt_catalog(data_dir=data_dir)
        models_data = load_json(data_dir / "models.json", {"models": []})
        policy_prompt_options, _, _ = deps.runtime_policy_service.load_policy_prompts_for_request(
            request=request,
            response=None,
            explicit_session_id=None,
            normalize_base_url=deps.normalize_base_url,
        )
        model_cfg = next(
            (m for m in models_data.get("models", []) if m["id"] == req.model_id),
            None,
        )
        hf_id = model_cfg.get("hf_id") if model_cfg else None
        token_counter: PromptTokenCounter = request.app.state.prompt_token_counter

        if schema_version == 3:
            raw_dynamic = resolve_dynamic_prompt_sections(
                req,
                prompts,
                policy_options=policy_prompt_options,
                strict=False,
            )
            resolved_dynamic = resolve_dynamic_prompt_variants(raw_dynamic)
            compiled = build_dynamic_prompt(
                resolved_dynamic,
                expand_placeholders=False,
            )
            token_counts = token_counter.count_dynamic_prompt_sections(
                hf_id=hf_id,
                sections=resolved_dynamic,
                compiled_prompt=compiled,
            )
            return {
                "compiled_prompt": compiled,
                "token_counts": token_counts,
            }

        raw_v2 = resolve_structured_prompt_sections(
            req,
            prompts,
            policy_options=policy_prompt_options,
            strict=False,
        )
        resolved_v2 = resolve_structured_prompt_variants(raw_v2)
        compiled = build_structured_prompt(
            resolved_v2,
            expand_placeholders=False,
        )
        token_counts = token_counter.count_prompt_sections(
            hf_id=hf_id,
            subject_text=resolved_v2.get("subject", ""),
            setting_text=resolved_v2.get("setting", ""),
            details_text=resolved_v2.get("details", ""),
            lighting_text=resolved_v2.get("lighting", ""),
            atmosphere_text=resolved_v2.get("atmosphere", ""),
            compiled_prompt=compiled,
        )
        return {
            "compiled_prompt": compiled,
            "token_counts": token_counts,
        }

    return router
