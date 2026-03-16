"""Prompt request resolution for section-schema payloads."""

from __future__ import annotations

from fastapi import HTTPException

from pipeworks.api.models import GenerateRequest
from pipeworks.api.prompt_builder import SECTION_ORDER
from pipeworks.api.services.prompt_catalog import build_prompt_lookup

PROMPT_SECTION_ORDER = SECTION_ORDER


def ensure_prompt_schema_v2(req: GenerateRequest) -> None:
    """Require explicit schema-v2 payloads for prompt compile/generate APIs."""
    if req.prompt_schema_version != 2:
        raise HTTPException(
            status_code=400,
            detail="prompt_schema_version=2 is required.",
        )


def resolve_structured_prompt_sections(
    req: GenerateRequest,
    prompts: dict,
    *,
    policy_options: list[dict] | None = None,
    strict: bool = False,
) -> dict[str, str]:
    """Resolve Subject/Setting/Details/Lighting/Atmosphere values from request."""
    prompt_lookup = build_prompt_lookup(prompts, policy_options)
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
