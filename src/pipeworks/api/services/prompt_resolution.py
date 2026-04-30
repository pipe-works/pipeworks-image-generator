"""Prompt request resolution for section-schema payloads."""

from __future__ import annotations

from fastapi import HTTPException

from pipeworks.api.models import GenerateRequest
from pipeworks.api.prompt_builder import SECTION_ORDER
from pipeworks.api.services.prompt_catalog import build_prompt_lookup

PROMPT_SECTION_ORDER = SECTION_ORDER

SUPPORTED_PROMPT_SCHEMA_VERSIONS = (2, 3)


def ensure_prompt_schema_v2(req: GenerateRequest) -> None:
    """Require explicit schema-v2 payloads for prompt compile/generate APIs."""
    if req.prompt_schema_version != 2:
        raise HTTPException(
            status_code=400,
            detail="prompt_schema_version=2 is required.",
        )


def ensure_prompt_schema(req: GenerateRequest) -> int:
    """Validate the request carries a supported prompt schema version.

    Returns the resolved schema version for downstream branching.
    """
    if req.prompt_schema_version not in SUPPORTED_PROMPT_SCHEMA_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "prompt_schema_version must be one of " f"{list(SUPPORTED_PROMPT_SCHEMA_VERSIONS)}."
            ),
        )
    return req.prompt_schema_version


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


def resolve_dynamic_prompt_sections(
    req: GenerateRequest,
    prompts: dict,
    *,
    policy_options: list[dict] | None = None,
    strict: bool = False,
) -> list[dict[str, str]]:
    """Resolve dynamic v3 sections to ordered ``[{label, text}, ...]`` list.

    Each section keeps the curator-supplied label so the compiled prompt
    can use it as a block header. Sections that resolve to empty strings
    are kept in the list with an empty ``text`` so token counting stays
    1:1 with the submitted slots; the builder is responsible for skipping
    empty entries when assembling the final prompt.
    """
    prompt_lookup = build_prompt_lookup(prompts, policy_options)
    sections = req.sections or []
    resolved: list[dict[str, str]] = []

    for section in sections:
        label = (section.label or "").strip() or "Policy"
        manual_value = (section.manual_text or "").strip()
        prompt_id = section.automated_prompt_id

        if section.mode == "automated" and prompt_id and prompt_id != "none":
            prompt = prompt_lookup.get(prompt_id)
            if prompt:
                text = (prompt.get("value") or "").strip()
            elif strict:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown automated prompt: {prompt_id}",
                )
            else:
                text = ""
        else:
            text = manual_value

        resolved.append({"label": label, "text": text})

    return resolved
