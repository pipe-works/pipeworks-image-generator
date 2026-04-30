"""Prompt request resolution for the dynamic-section schema (v3)."""

from __future__ import annotations

from fastapi import HTTPException

from pipeworks.api.models import GenerateRequest
from pipeworks.api.services.prompt_catalog import build_prompt_lookup


def ensure_prompt_schema(req: GenerateRequest) -> None:
    """Reject any request whose schema version is not the current one (3)."""
    if req.prompt_schema_version != 3:
        raise HTTPException(
            status_code=400,
            detail="prompt_schema_version=3 is required.",
        )


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
