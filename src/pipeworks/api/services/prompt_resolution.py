"""Prompt request resolution for legacy and section-schema payloads."""

from __future__ import annotations

from fastapi import HTTPException

from pipeworks.api.models import GenerateRequest
from pipeworks.api.prompt_builder import SECTION_ORDER
from pipeworks.api.services.prompt_catalog import build_prompt_lookup

PROMPT_SECTION_ORDER = SECTION_ORDER


def resolve_prompt_parts(
    req: GenerateRequest,
    prompts: dict,
    *,
    strict: bool = False,
) -> tuple[str, str, str]:
    """Resolve prepend, main scene, and append values from request."""
    prompt_lookup = {
        prompt["id"]: prompt for prompt in prompts.get("all_prompts", []) if prompt.get("id")
    }

    prepend_value = ""
    if req.prepend_mode == "manual":
        prepend_value = (req.manual_prepend or "").strip()
    elif req.prepend_prompt_id and req.prepend_prompt_id != "none":
        prepend = prompt_lookup.get(req.prepend_prompt_id)
        if prepend:
            prepend_value = prepend["value"]

    main_scene = ""
    if req.prompt_mode == "manual":
        main_scene = (req.manual_prompt or "").strip()
    elif req.prompt_mode == "automated":
        if req.automated_prompt_id and req.automated_prompt_id != "none":
            automated = prompt_lookup.get(req.automated_prompt_id)
            if automated:
                main_scene = automated["value"]
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

    append_value = ""
    if req.append_mode == "manual":
        append_value = (req.manual_append or "").strip()
    elif req.append_prompt_id and req.append_prompt_id != "none":
        append = prompt_lookup.get(req.append_prompt_id)
        if append:
            append_value = append["value"]

    return prepend_value, main_scene, append_value


def request_uses_section_schema(req: GenerateRequest) -> bool:
    """Return True when request includes five-section composer fields."""
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
