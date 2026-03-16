"""Prompt catalog loading helpers and legacy fallback support."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeworks.api.services.deprecations import (
    LEGACY_PROMPTS_JSON_FALLBACK_WARNING,
    warn_once,
)
from pipeworks.core.model_manager import get_model_runtime_support

logger = logging.getLogger(__name__)


def load_json(path: Path, default):
    """Load JSON from disk, returning ``default`` for missing/invalid files."""
    if path.exists():
        try:
            with open(path) as handle:
                return json.load(handle)
        except Exception:
            return default
    return default


def _load_prompt_list(
    path: Path,
    *,
    legacy_prompts: dict,
    legacy_key: str,
) -> tuple[list[dict], bool]:
    """Load one prompt-library file with optional legacy fallback support."""
    data = load_json(path, None)

    prompts: list[dict] = []
    used_legacy_fallback = False
    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict):
        prompts = data.get("prompts", [])

    if not prompts:
        prompts = legacy_prompts.get(legacy_key, [])
        used_legacy_fallback = bool(prompts)

    normalized: list[dict] = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        item = dict(prompt)
        item.setdefault("source_section", path.stem)
        normalized.append(item)

    return normalized, used_legacy_fallback


def _merge_prompt_lists(*prompt_lists: list[dict]) -> list[dict]:
    """Merge prompt lists while preserving order and de-duplicating by id."""
    merged: list[dict] = []
    seen_ids: set[str] = set()

    for prompt_list in prompt_lists:
        for prompt in prompt_list:
            prompt_id = prompt.get("id")
            if not prompt_id:
                continue
            if prompt_id in seen_ids:
                logger.warning(
                    "Duplicate prompt id '%s' detected; keeping first occurrence.", prompt_id
                )
                continue
            seen_ids.add(prompt_id)
            merged.append(prompt)

    return merged


def annotate_models_with_runtime_support(models: list[dict]) -> list[dict]:
    """Attach runtime availability metadata to model config entries."""
    annotated: list[dict] = []
    for model in models:
        item = dict(model)
        is_available, reason = get_model_runtime_support(item.get("hf_id", ""))
        item["is_available"] = is_available
        item["unavailable_reason"] = reason
        annotated.append(item)
    return annotated


def load_prompt_catalog(*, data_dir: Path) -> dict:
    """Load split prompt files and merged selector catalog."""
    legacy_prompts = load_json(data_dir / "prompts.json", {})

    prepend_library, prepend_legacy = _load_prompt_list(
        data_dir / "prepend.json",
        legacy_prompts=legacy_prompts,
        legacy_key="prepend_prompts",
    )
    main_library, main_legacy = _load_prompt_list(
        data_dir / "main.json",
        legacy_prompts=legacy_prompts,
        legacy_key="automated_prompts",
    )
    append_library, append_legacy = _load_prompt_list(
        data_dir / "append.json",
        legacy_prompts=legacy_prompts,
        legacy_key="append_prompts",
    )

    if prepend_legacy or main_legacy or append_legacy:
        warn_once(
            logger=logger,
            key="legacy-prompts-json-fallback",
            message=LEGACY_PROMPTS_JSON_FALLBACK_WARNING,
        )

    merged_prompts = _merge_prompt_lists(prepend_library, main_library, append_library)

    return {
        "prepend_library": prepend_library,
        "main_library": main_library,
        "append_library": append_library,
        "all_prompts": merged_prompts,
        "prepend_prompts": merged_prompts,
        "automated_prompts": merged_prompts,
        "append_prompts": merged_prompts,
    }


def build_prompt_lookup(
    prompts: dict,
    policy_options: list[dict] | None = None,
) -> dict[str, dict]:
    """Build prompt lookup map from prompt libraries and policy snippets."""
    lookup: dict[str, dict] = {}
    for prompt in prompts.get("all_prompts", []):
        prompt_id = prompt.get("id")
        if not prompt_id:
            continue
        lookup[prompt_id] = prompt

    for option in policy_options or []:
        option_id = option.get("id")
        if not option_id or option_id in lookup:
            continue
        lookup[option_id] = option

    return lookup
