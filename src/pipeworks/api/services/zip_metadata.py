"""Zip metadata helpers for gallery exports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pipeworks.api.runtime_mode import get_runtime_mode
from pipeworks.api.services.prompt_catalog import (
    build_prompt_lookup,
    load_json,
    load_prompt_catalog,
)
from pipeworks.api.services.prompt_resolution import PROMPT_SECTION_ORDER
from pipeworks.api.services.runtime_policy import RuntimePolicyService


def build_zip_metadata_for_entry(
    *,
    entry: dict,
    data_dir: Path,
    runtime_policy_service: RuntimePolicyService,
    normalize_base_url,
) -> dict:
    """Build structured metadata dictionary for a gallery entry zip."""
    image_id = entry["id"]

    models_data = load_json(data_dir / "models.json", {"models": []})
    model_cfg = next(
        (m for m in models_data.get("models", []) if m["id"] == entry.get("model_id")),
        None,
    )

    created_at_ts = entry.get("created_at")
    created_at_iso = (
        datetime.fromtimestamp(created_at_ts, tz=UTC).isoformat() if created_at_ts else None
    )

    prompts = load_prompt_catalog(data_dir=data_dir)
    runtime_state = get_runtime_mode()
    policy_options = runtime_policy_service.load_policy_prompt_options(
        active_server_url=runtime_state.active_server_url,
        session_id=None,
        normalize_base_url=normalize_base_url,
    )
    prompt_lookup = build_prompt_lookup(prompts, policy_options)

    prepend_mode = entry.get("prepend_mode", "template")
    append_mode = entry.get("append_mode", "template")

    prepend_id = entry.get("prepend_prompt_id")
    prepend_preset = prompt_lookup.get(prepend_id, {}) if prepend_id else {}
    if prepend_mode == "manual":
        prepend_text = entry.get("manual_prepend", "")
    else:
        prepend_text = prepend_preset.get("value", "")

    prompt_mode = entry.get("prompt_mode", "manual")
    automated_id = entry.get("automated_prompt_id")
    automated_preset = prompt_lookup.get(automated_id, {}) if automated_id else {}
    if prompt_mode == "manual":
        main_text = entry.get("manual_prompt", "") or ""
    else:
        main_text = automated_preset.get("value", "")

    append_id = entry.get("append_prompt_id")
    append_preset = prompt_lookup.get(append_id, {}) if append_id else {}
    if append_mode == "manual":
        append_text = entry.get("manual_append", "")
    else:
        append_text = append_preset.get("value", "")

    sections_metadata: dict[str, dict] = {}
    for section in PROMPT_SECTION_ORDER:
        section_mode = entry.get(f"{section}_mode", "manual")
        section_id = entry.get(f"automated_{section}_prompt_id")
        section_preset = prompt_lookup.get(section_id, {}) if section_id else {}
        section_text = (entry.get(f"manual_{section}") or "").strip()
        if not section_text and section_mode == "automated":
            section_text = (section_preset.get("value") or "").strip()
        sections_metadata[section] = {
            "mode": section_mode,
            "preset_id": section_id,
            "preset_label": section_preset.get("label"),
            "text": section_text,
        }

    return {
        "id": image_id,
        "model": {
            "id": entry.get("model_id"),
            "label": model_cfg["label"] if model_cfg else entry.get("model_label"),
        },
        "prompt": {
            "compiled": entry.get("compiled_prompt", ""),
            "schema_version": entry.get("prompt_schema_version", 1),
            "sections": sections_metadata,
            "prepend": {
                "mode": prepend_mode,
                "preset_id": prepend_id,
                "preset_label": prepend_preset.get("label"),
                "text": prepend_text,
            },
            "main": {
                "mode": prompt_mode,
                "preset_id": automated_id,
                "preset_label": automated_preset.get("label"),
                "text": main_text,
            },
            "append": {
                "mode": append_mode,
                "preset_id": append_id,
                "preset_label": append_preset.get("label"),
                "text": append_text,
            },
        },
        "generation": {
            "width": entry.get("width"),
            "height": entry.get("height"),
            "aspect_ratio": entry.get("aspect_ratio_id"),
            "steps": entry.get("steps"),
            "guidance": entry.get("guidance"),
            "seed": entry.get("seed"),
            "negative_prompt": entry.get("negative_prompt"),
            "scheduler": entry.get("scheduler"),
        },
        "batch": {
            "index": entry.get("batch_index"),
            "size": entry.get("batch_size"),
            "seed": entry.get("batch_seed"),
        },
        "created_at": created_at_iso,
        "is_favourite": entry.get("is_favourite", False),
    }
