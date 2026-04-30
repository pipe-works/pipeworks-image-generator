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
from pipeworks.api.services.runtime_policy import RuntimePolicyService


def build_zip_metadata_for_entry(
    *,
    entry: dict,
    data_dir: Path,
    runtime_policy_service: RuntimePolicyService,
    normalize_base_url,
) -> dict:
    """Build structured metadata dictionary for a gallery entry zip.

    Reads dynamic-section (schema v3) gallery entries. Older entries that
    were written under the legacy v1/v2 schemas surface only as their
    compiled prompt — section-level breakdown is unavailable for them.
    """
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

    sections_metadata: list[dict] = []
    for section in entry.get("sections") or []:
        if not isinstance(section, dict):
            continue
        mode = section.get("mode", "manual")
        preset_id = section.get("automated_prompt_id")
        preset = prompt_lookup.get(preset_id, {}) if preset_id else {}
        manual_text = (section.get("manual_text") or "").strip()
        if not manual_text and mode == "automated":
            text = (preset.get("value") or "").strip()
        else:
            text = manual_text
        sections_metadata.append(
            {
                "label": section.get("label", "Policy"),
                "mode": mode,
                "preset_id": preset_id,
                "preset_label": preset.get("label"),
                "text": text,
            }
        )

    return {
        "id": image_id,
        "model": {
            "id": entry.get("model_id"),
            "label": model_cfg["label"] if model_cfg else entry.get("model_label"),
        },
        "prompt": {
            "compiled": entry.get("compiled_prompt", ""),
            "schema_version": entry.get("prompt_schema_version", 3),
            "sections": sections_metadata,
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
